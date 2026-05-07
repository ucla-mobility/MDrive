#!/usr/bin/env python3
"""One-command launcher for the HITL stack: CARLA + dashboard (+ optional eval).

Default mode brings up the two long-running services the operator needs:
a CARLA server and the dashboard backend.

Adding `--eval-route <dir>` ALSO spawns `tools/run_custom_eval.py` against
that route with `COLMDRIVER_HITL=1` set, so a single command brings the
whole stack up end-to-end.

Run:
    conda activate colmdrivermarco2

    # Just the services (CARLA + dashboard); kick the eval off yourself.
    python -m tools.hitl_run

    # Full stack: services + scenario runner (one command, end-to-end).
    python -m tools.hitl_run \
        --eval-route scenarioset/interaction/Highway_On-Ramp_Merge/1

Default ports: CARLA 4010 (TM 4015), dashboard 8765.

From your laptop, forward the dashboard:
    ssh -L 8765:localhost:8765 <eval-server>
    # then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _check_port_free(host: str, port: int) -> Optional[str]:
    """Return None if `host:port` can be bound to right now, else a reason str.

    Catches the most common boot failure: a stale dashboard from a previous
    `hitl_run` is still holding 8765 and uvicorn refuses to bind, but the
    user just sees "[hitl_run] dashboard exited immediately" and gives up.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        s.close()
        return None
    except OSError as exc:
        s.close()
        return f"{host}:{port} is in use ({exc!s})"


def _post_dashboard(host: str, port: int, path: str, body: dict, timeout_s: float = 3.0) -> Optional[dict]:
    from urllib import request as _req
    import json as _json
    url = f"http://{host}:{port}{path}"
    data = _json.dumps(body).encode("utf-8")
    req = _req.Request(url, data=data, method="POST",
                       headers={"content-type": "application/json"})
    with _req.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return _json.loads(raw.decode("utf-8") or "{}")


def _get_dashboard(host: str, port: int, path: str, timeout_s: float = 2.0) -> Optional[dict]:
    from urllib import request as _req
    from urllib import error as _err
    import json as _json
    url = f"http://{host}:{port}{path}"
    try:
        with _req.urlopen(url, timeout=timeout_s) as resp:
            raw = resp.read()
        return _json.loads(raw.decode("utf-8") or "{}")
    except (_err.URLError, OSError, _json.JSONDecodeError):
        return None


def _fetch_scenario_summary(host: str, port: int, scenario, scenario_dir) -> dict:
    """Pull the current results snapshot from the dashboard and pin it as
    the per-scenario record for the experiment summary."""
    snap = _get_dashboard(host, port, "/api/results") or {}
    egos_summary: dict = {}
    aggregate_inf: dict = {}
    for ego_id_str, ego in (snap.get("egos") or {}).items():
        infractions = ego.get("infractions") or {}
        egos_summary[ego_id_str] = {
            "ds_pct": ego.get("ds_pct"),
            "rc_pct": ego.get("rc_pct"),
            "score_penalty": ego.get("score_penalty"),
            "scores_provenance": ego.get("scores_provenance"),
            "speed_kmh_final": ego.get("speed_kmh"),
            "distance_to_goal_m_final": ego.get("distance_to_goal_m"),
            "completed": ego.get("completed"),
            "terminal_state": ego.get("terminal_state"),
            "terminal_reason": ego.get("terminal_reason"),
            "infractions": infractions,
        }
        for k, v in infractions.items():
            try:
                aggregate_inf[k] = aggregate_inf.get(k, 0) + int(v)
            except Exception:
                pass
    return {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "instance": scenario.instance,
        "town": scenario.town,
        "ego_count": scenario.ego_count,
        "results_dir": str(scenario_dir) if scenario_dir else None,
        "step_final": snap.get("step"),
        "sim_time_s_final": snap.get("sim_time_s"),
        "egos": egos_summary,
        "infractions_aggregate": aggregate_inf,
    }


def _write_experiment_summary(experiment_dir: "Path", experiment_id: str,
                              queue: list, results: list) -> None:
    import json as _json
    summary = {
        "experiment_id": experiment_id,
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios_total": len(queue),
        "scenarios_completed": len(results),
        "scenarios": results,
    }
    # Aggregate per-category averages of DS/RC for quick eyeballing.
    by_cat: dict = {}
    for r in results:
        cat = r.get("category") or "unknown"
        by_cat.setdefault(cat, []).append(r)
    averages: dict = {}
    for cat, rs in by_cat.items():
        ds_vals, rc_vals = [], []
        for r in rs:
            for ego in (r.get("egos") or {}).values():
                if ego.get("ds_pct") is not None:
                    ds_vals.append(float(ego["ds_pct"]))
                if ego.get("rc_pct") is not None:
                    rc_vals.append(float(ego["rc_pct"]))
        averages[cat] = {
            "n_egos": len(ds_vals),
            "ds_pct_mean": (sum(ds_vals) / len(ds_vals)) if ds_vals else None,
            "rc_pct_mean": (sum(rc_vals) / len(rc_vals)) if rc_vals else None,
        }
    summary["averages_by_category"] = averages
    path = experiment_dir / "summary.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        _json.dump(summary, fh, default=str, indent=2)
    tmp.replace(path)


def _wait_dashboard_responds(host: str, port: int, timeout_s: float = 8.0) -> Optional[str]:
    """Poll http://host:port/api/state until 200 or timeout. Returns None on
    success, an error string on failure. Confirms uvicorn isn't just bound
    but actually serving."""
    from urllib import request as _req
    from urllib import error as _err

    deadline = time.monotonic() + timeout_s
    last_err = "no attempt yet"
    url = f"http://{host}:{port}/api/state"
    while time.monotonic() < deadline:
        try:
            with _req.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return None
                last_err = f"HTTP {resp.status}"
        except (_err.URLError, OSError, ConnectionError) as exc:
            last_err = str(exc)
        time.sleep(0.25)
    return f"never returned 200 on /api/state (last error: {last_err})"


_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
for _p in (_REPO_ROOT, _TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hitl_run",
        description="Start CARLA + the HITL dashboard together.",
    )
    # CARLA
    p.add_argument("--carla-host", default="127.0.0.1")
    p.add_argument("--carla-port", type=int, default=4010)
    p.add_argument("--tm-port-offset", type=int, default=5)
    p.add_argument(
        "--carla-arg",
        action="append",
        default=None,
        help=(
            "Extra arg passed to CarlaUE4.sh (repeatable). Default: "
            "-RenderOffScreen. Note: -quality-level=Low SEGFAULTS CARLA "
            "0.9.12 during scenario teardown (Low skips asset loading the "
            "sim later dereferences); leave it off."
        ),
    )
    p.add_argument(
        "--carla-root",
        default=None,
        help="Path to CARLA install. Defaults to $CARLA_ROOT or <repo>/carla912.",
    )
    p.add_argument("--ready-timeout", type=float, default=60.0)
    p.add_argument("--port-tries", type=int, default=1)
    p.add_argument("--port-step", type=int, default=10)
    p.add_argument(
        "--no-carla",
        action="store_true",
        help="Skip CARLA launch entirely (useful if CARLA is already running).",
    )
    p.add_argument(
        "--carla-existing",
        action="store_true",
        help=(
            "Don't launch CARLA, but DO probe the requested port to confirm "
            "an already-running CARLA is reachable."
        ),
    )

    # Dashboard
    p.add_argument("--dashboard-host", default="127.0.0.1")
    p.add_argument("--dashboard-port", type=int, default=8765)
    p.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip dashboard launch (rare — only useful for CARLA-only tests).",
    )
    p.add_argument(
        "--decision-log",
        default=None,
        help=(
            "Append-only JSONL path for human decisions (passed to the "
            "dashboard). Defaults to runs/<route-name>/hitl_decisions.jsonl "
            "when --eval-route is set, otherwise no file logging."
        ),
    )

    # Eval (optional — when set, also spawn run_custom_eval.py with HITL on).
    p.add_argument(
        "--eval-route",
        action="append",
        default=None,
        help=(
            "Routes directory to run with COLMDRIVER_HITL=1 (e.g. "
            "scenarioset/interaction/Highway_On-Ramp_Merge/1). Repeatable — pass "
            "multiple to run them sequentially. CARLA stays alive between "
            "scenarios."
        ),
    )
    p.add_argument(
        "--scenarios",
        action="append",
        default=None,
        help=(
            "Scenario specs resolved against scenarioset/interaction. Examples: "
            "'all', 'Highway_On-Ramp_Merge', 'Highway_On-Ramp_Merge/1', "
            "'Highway_On-Ramp_Merge/*', or a glob path. Repeatable."
        ),
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print every discovered scenario and exit.",
    )
    p.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "Experiment label. All results land under "
            "<experiment-root>/<experiment-id>/. Defaults to a timestamp."
        ),
    )
    p.add_argument(
        "--experiment-root",
        default="runs",
        help=(
            "Root directory for experiment dirs. Default 'runs'. Use e.g. "
            "'results/humanexperiments' to keep human-study runs separate "
            "from automated eval runs."
        ),
    )
    p.add_argument(
        "--eval-arg",
        action="append",
        default=None,
        help="Extra arg passed through to run_custom_eval.py (repeatable).",
    )
    p.add_argument(
        "--hitl-timeout-s",
        type=float,
        default=30.0,
        help=(
            "Per-negotiation human-input timeout. Sets "
            "COLMDRIVER_HITL_TIMEOUT_S for the eval subprocess. Default 30s."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("direct", "negotiation"),
        default="direct",
        help=(
            "HITL mode. 'direct' = thin agent, dashboard drives every tick "
            "via hold-last decisions (fast, ~10 Hz ticks). "
            "'negotiation' = full CoLMDriver, human only replaces LLM "
            "negotiation decisions (slow, faithful). Default: direct."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Redirect CARLA's stdout/stderr and the eval subprocess's "
            "stdout/stderr to log files in the results dir, so the "
            "hitl_run terminal stays clean. Logs land at "
            "<results-dir>/{carla.log,eval.log}."
        ),
    )

    return p.parse_args(argv)


def _spawn_dashboard(
    host: str,
    port: int,
    decision_log: Optional[Path],
    results_dir: Optional[Path],
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "tools.hitl_dashboard",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if results_dir is not None:
        cmd += ["--results-dir", str(results_dir)]
    if decision_log is not None:
        cmd += ["--decision-log", str(decision_log)]
    print(f"[hitl_run] starting dashboard: {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(_REPO_ROOT))


def _spawn_eval(
    *,
    route_dir: Path,
    extra_args: List[str],
    carla_port: int,
    dashboard_url: str,
    hitl_timeout_s: float,
    mode: str,
    log_path: Optional[Path] = None,
) -> subprocess.Popen:
    """Spawn `tools/run_custom_eval.py` with HITL env wired up.

    `mode == "direct"`     : load `team_code/hitl_agent.py` directly. No
                             CoLMDriver perception/planning. Sub-second ticks.
    `mode == "negotiation"`: leave the planner default (`colmdriver`) and set
                             COLMDRIVER_HITL=1 so only LLM-negotiation
                             decisions are replaced by the dashboard.
    """
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "tools" / "run_custom_eval.py"),
        "--routes-dir",
        str(route_dir),
        "--port",
        str(carla_port),
        "--no-start-carla",
        # HITL is interactive — every click of Start (or Retry) MUST run a
        # fresh eval. run_custom_eval defaults to --skip-existed=on, which
        # checks SAVE_PATH for any prior valid run of this route and
        # silently returns without spawning the leaderboard. That would
        # leave the dashboard stuck on "Waiting for the agent…" while the
        # launcher cycles through the queue.
        "--no-skip-existed",
    ]
    env = os.environ.copy()
    env["COLMDRIVER_HITL_URL"] = dashboard_url
    env["COLMDRIVER_HITL_TIMEOUT_S"] = str(hitl_timeout_s)

    if mode == "direct":
        # Override --agent to the thin direct agent. Still goes through
        # run_custom_eval so route prep / scenario setup is unchanged.
        cmd += [
            "--agent",
            str(_REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "hitl_agent.py"),
        ]
    elif mode == "negotiation":
        env["COLMDRIVER_HITL"] = "1"
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    cmd += list(extra_args)
    print(f"[hitl_run] starting eval (mode={mode}): {' '.join(cmd)}")
    print(
        f"[hitl_run]   COLMDRIVER_HITL_URL={dashboard_url}"
        + ("  COLMDRIVER_HITL=1" if mode == "negotiation" else "")
    )
    stdout = stderr = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a", buffering=1)  # line-buffered
        stdout = log_fp
        stderr = subprocess.STDOUT
        print(f"[hitl_run]   eval stdout/stderr → {log_path}")
    return subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env, stdout=stdout, stderr=stderr)


def _ensure_carla_alive(handle, args, extra_args, experiment_dir):
    """Return a live CarlaHandle. If the current one is dead, restart CARLA.

    Returns None only when restart fails — caller should abort the queue.
    """
    from tools.hitl_start_carla import start_carla, wait_for_carla_ready
    # Cheap process-level liveness check first (no RPC round-trip).
    proc = getattr(getattr(handle, "manager", None), "process", None)
    proc_alive = proc is None or (proc.poll() is None)
    rpc_ok = False
    if proc_alive:
        # Tight RPC probe — if it succeeds quickly we're done.
        try:
            wait_for_carla_ready(
                host=handle.host,
                port=handle.port,
                timeout_s=8.0,
                client_timeout_s=4.0,
                retry_interval_s=1.0,
                carla_root=str(handle.carla_root),
            )
            rpc_ok = True
        except Exception as exc:
            print(f"[hitl_run] CARLA RPC probe failed ({exc!r}); restart needed.",
                  file=sys.stderr)
    else:
        rc = proc.returncode
        print(f"[hitl_run] CARLA process exited (rc={rc}); restart needed.",
              file=sys.stderr)

    if rpc_ok:
        return handle

    # Restart. Best-effort stop the old handle, then start a fresh one on
    # a *new* port so any TIME_WAIT sockets from the dead CARLA don't
    # collide on bind. start_carla's port_tries already handles bumping
    # if the requested port is taken.
    try:
        handle.stop(timeout=10.0)
    except Exception:
        pass
    new_port = handle.port + max(args.port_step, 10)
    carla_log_path = None
    if args.quiet and experiment_dir is not None:
        carla_log_path = str(experiment_dir / f"carla.restart_{int(time.time())}.log")
    print(f"[hitl_run] restarting CARLA on port {new_port}…")
    try:
        new_handle = start_carla(
            host=args.carla_host,
            port=new_port,
            tm_port_offset=args.tm_port_offset,
            extra_args=extra_args,
            carla_root=args.carla_root,
            port_tries=max(args.port_tries, 4),
            port_step=args.port_step,
            quiet=args.quiet,
            server_log_path=carla_log_path,
        )
        wait_for_carla_ready(
            host=new_handle.host,
            port=new_handle.port,
            timeout_s=args.ready_timeout,
            carla_root=str(new_handle.carla_root),
        )
        print(f"[hitl_run] CARLA restarted at port={new_handle.port}, tm_port={new_handle.tm_port}")
        return new_handle
    except Exception as exc:
        print(f"[hitl_run] CARLA restart FAILED: {exc!r}", file=sys.stderr)
        return None


def _stop_dashboard(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    print("[hitl_run] stopping dashboard...")
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[hitl_run] dashboard did not exit, killing.")
        proc.kill()
        proc.wait(timeout=timeout)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # --- Resolve the scenario queue first ----------------------------------
    from tools.hitl_scenarios import resolve_scenarios, list_scenarios, _format_table

    if args.list_scenarios:
        scenarios = list_scenarios()
        print(f"Found {len(scenarios)} scenario(s):")
        print(_format_table(scenarios))
        return 0

    queue = []
    raw_specs: List[str] = []
    if args.eval_route:
        raw_specs.extend(args.eval_route)
    if args.scenarios:
        raw_specs.extend(args.scenarios)
    if raw_specs:
        queue = resolve_scenarios(raw_specs)
        if not queue:
            print(
                f"[hitl_run] no scenarios matched specs={raw_specs}.\n"
                "  Run `python -m tools.hitl_run --list-scenarios` to see "
                "what's available.",
                file=sys.stderr,
            )
            return 7
        print(f"[hitl_run] queued {len(queue)} scenario(s):")
        for i, s in enumerate(queue, 1):
            print(f"  {i:>3}. {s.display}  (egos={s.ego_count}, town={s.town})")

    # Default CARLA flags for HITL: headless only. -quality-level=Low used
    # to be in here too but on CARLA 0.9.12 the Low preset triggered a
    # SIGSEGV during scenario route_start_gate teardown — UE4 dereferenced
    # assets that the Low preset hadn't loaded. Stick with the default
    # (Epic) preset; it's stable. Override via --carla-arg if needed.
    extra_args = list(args.carla_arg) if args.carla_arg is not None else [
        "-RenderOffScreen",
    ]

    # Lazy import so `--help` works even if CARLA / fastapi aren't ready.
    from tools.hitl_start_carla import start_carla, wait_for_carla_ready, CarlaHandle

    handle: Optional[CarlaHandle] = None
    dash_proc: Optional[subprocess.Popen] = None
    eval_proc: Optional[subprocess.Popen] = None

    # --- Experiment dir layout --------------------------------------------
    experiment_id = args.experiment_id or time.strftime("%Y%m%d_%H%M%S")
    experiment_dir: Optional[Path] = None
    if queue:
        root_arg = Path(args.experiment_root).expanduser()
        if not root_arg.is_absolute():
            root_arg = _REPO_ROOT / root_arg
        experiment_dir = (root_arg / experiment_id).resolve()
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"[hitl_run] experiment dir: {experiment_dir}")

    # The decision-log override remains supported (single-file logging),
    # but the per-scenario bundle is the canonical layout for queues.
    decision_log: Optional[Path] = (
        Path(args.decision_log).expanduser().resolve()
        if args.decision_log else None
    )

    stop_requested = {"flag": False}

    def _on_signal(signum, frame):  # noqa: ARG001
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        # 1. CARLA
        if args.no_carla:
            print("[hitl_run] --no-carla: skipping CARLA entirely.")
        elif args.carla_existing:
            print(
                f"[hitl_run] --carla-existing: probing existing CARLA at "
                f"{args.carla_host}:{args.carla_port} ..."
            )
            wait_for_carla_ready(
                host=args.carla_host,
                port=args.carla_port,
                timeout_s=args.ready_timeout,
                carla_root=args.carla_root,
            )
        else:
            carla_log_path = None
            if args.quiet and experiment_dir is not None:
                carla_log_path = str(experiment_dir / "carla.log")
            print(
                f"[hitl_run] launching CARLA at {args.carla_host}:{args.carla_port} "
                f"(tm offset={args.tm_port_offset}, extra={extra_args}"
                + (f", log={carla_log_path}" if carla_log_path else "")
                + ")"
            )
            handle = start_carla(
                host=args.carla_host,
                port=args.carla_port,
                tm_port_offset=args.tm_port_offset,
                extra_args=extra_args,
                carla_root=args.carla_root,
                port_tries=args.port_tries,
                port_step=args.port_step,
                quiet=args.quiet,
                server_log_path=carla_log_path,
            )
            print(
                f"[hitl_run] CARLA up: port={handle.port}, tm_port={handle.tm_port}"
            )
            wait_for_carla_ready(
                host=handle.host,
                port=handle.port,
                timeout_s=args.ready_timeout,
                carla_root=str(handle.carla_root),
            )

        # Compute the dashboard's *initial* results dir. For a queue, each
        # scenario gets its own subdir, retargeted via /api/results_dir
        # before each spawn.
        initial_results_dir = experiment_dir
        # 2. Dashboard
        if not args.no_dashboard:
            # Pre-flight: refuse to start if the port is already taken.
            # Without this we'd silently spawn uvicorn, see it crash with
            # "Address already in use", and the operator's browser would
            # never load. The most common cause is a stale dashboard from
            # a previous hitl_run run.
            busy = _check_port_free(args.dashboard_host, args.dashboard_port)
            if busy:
                print(
                    f"[hitl_run] dashboard port unavailable: {busy}\n"
                    f"  Find the offender: ss -lntp | grep :{args.dashboard_port}\n"
                    f"  Or pick a different port: --dashboard-port <N>\n"
                    f"  Or kill stale dashboards: pkill -f 'hitl_dashboard'",
                    file=sys.stderr,
                )
                return 6
            dash_proc = _spawn_dashboard(
                args.dashboard_host, args.dashboard_port, decision_log, initial_results_dir
            )
            # Give uvicorn a moment to bind.
            time.sleep(1.0)
            if dash_proc.poll() is not None:
                rc = dash_proc.returncode
                print(f"[hitl_run] dashboard exited immediately (rc={rc}). Aborting.")
                return 4

            # Active reachability probe: confirm /api/state actually
            # responds. Catches the case where uvicorn says "Application
            # startup complete" but the bind interface or some middleware
            # is wrong.
            err = _wait_dashboard_responds(
                args.dashboard_host, args.dashboard_port, timeout_s=8.0
            )
            if err is not None:
                print(
                    f"[hitl_run] WARN dashboard does NOT respond on the "
                    f"server itself: {err}\n"
                    f"  This means the operator's browser will never load "
                    "the page either.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[hitl_run] dashboard verified responding at "
                    f"{args.dashboard_host}:{args.dashboard_port}/api/state"
                )

        # 3. Banner.
        print()
        print("=" * 72)
        print("[hitl_run] HITL stack is up.")
        if handle is not None:
            print(f"  CARLA      : {handle.host}:{handle.port}  (tm {handle.tm_port})")
        elif args.carla_existing:
            print(f"  CARLA      : {args.carla_host}:{args.carla_port}  (existing)")
        else:
            print("  CARLA      : (skipped)")
        if dash_proc is not None:
            host = socket.gethostname()
            print(f"  Dashboard  : bound on {args.dashboard_host}:{args.dashboard_port}")
            print(f"")
            print(f"  ┌── To open in your browser ──")
            if args.dashboard_host in ("0.0.0.0", "::"):
                print(
                    f"  │  Public bind. From any machine that can reach the "
                    f"server: http://{host}:{args.dashboard_port}"
                )
            else:
                print(f"  │  Bind is loopback-only ({args.dashboard_host}). Choose ONE:")
                print(f"  │")
                print(f"  │  (a) ON YOUR LAPTOP, open a tunnel:")
                print(f"  │      ssh -N -L {args.dashboard_port}:localhost:{args.dashboard_port} {os.environ.get('USER','user')}@{host}")
                print(f"  │      then open http://localhost:{args.dashboard_port}")
                print(f"  │")
                print(f"  │  (b) Re-run with --dashboard-host 0.0.0.0 to bind publicly,")
                print(f"  │      then open http://{host}:{args.dashboard_port} from your laptop")
                print(f"  │      (only safe on a trusted network).")
            print(f"  │")
            print(f"  │  All egos are held at brake=1 until your browser connects.")
            print(f"  └─────────────────────────────")
        if experiment_dir is not None:
            print(f"  Results    : {experiment_dir}/  (per-scenario subdirs + summary.json)")
        elif decision_log is not None:
            print(f"  Decisions  : {decision_log}")
        if queue:
            print(f"  Queue      : {len(queue)} scenarios")
        print("  Stop with Ctrl-C.")
        print("=" * 72)
        print()

        # 4. Multi-scenario loop. CARLA + dashboard stay up; only the eval
        # subprocess restarts per scenario.
        carla_port_for_eval = handle.port if handle is not None else args.carla_port
        dashboard_url = f"http://{args.dashboard_host}:{args.dashboard_port}"
        extra_eval_args = list(args.eval_arg) if args.eval_arg else []
        per_scenario_results: List[Dict[str, Any]] = []

        if not queue:
            # Run was invoked without --eval-route or --scenarios. Hold until
            # Ctrl-C just like the old behavior.
            while not stop_requested["flag"]:
                if dash_proc is not None and dash_proc.poll() is not None:
                    print(f"[hitl_run] dashboard died (rc={dash_proc.returncode}). Stopping.")
                    break
                time.sleep(0.5)
            print("[hitl_run] received stop signal.")
            return 0

        for idx, scenario in enumerate(queue, start=1):
            if stop_requested["flag"]:
                break
            scenario_dir = experiment_dir / scenario.scenario_id if experiment_dir else None
            attempt = 0
            while True:
                attempt += 1
                if scenario_dir is not None:
                    if attempt > 1:
                        # Retry: wipe everything from the previous attempt so
                        # the human gets a clean log. mkdir(exist_ok=True)
                        # would happily merge old + new files; we want a
                        # true fresh start.
                        try:
                            shutil.rmtree(scenario_dir)
                        except FileNotFoundError:
                            pass
                        except Exception as exc:
                            print(
                                f"[hitl_run] WARN couldn't wipe {scenario_dir} "
                                f"before retry: {exc!r}", file=sys.stderr,
                            )
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    # Tell the dashboard to retarget its results-dir AND
                    # reset per-ego in-memory state. /api/scenario_started
                    # also clears retry_scenario_pending and end_scenario_
                    # pending, ready=False (operator clicks Start again).
                    try:
                        _post_dashboard(args.dashboard_host, args.dashboard_port,
                                        "/api/results_dir", {"path": str(scenario_dir)})
                        _post_dashboard(args.dashboard_host, args.dashboard_port,
                                        "/api/scenario_started", {
                                            "scenario_id": scenario.scenario_id,
                                            "queue_index": idx,
                                            "queue_total": len(queue),
                                        })
                    except Exception as exc:
                        print(f"[hitl_run] WARN couldn't notify dashboard of scenario change: {exc!r}",
                              file=sys.stderr)

                # CARLA-alive check. UE4 segfaults during scenario teardown
                # are common in 0.9.12; a dead CARLA would make every
                # subsequent eval hang waiting for an RPC response that
                # never comes. Probe get_world() with a tight budget; if
                # it fails AND the launcher owns the CARLA handle, kill
                # and restart it before spawning the eval.
                if handle is not None and not args.no_carla and not args.carla_existing:
                    handle = _ensure_carla_alive(
                        handle, args, extra_args, experiment_dir,
                    )
                    if handle is None:
                        print("[hitl_run] CARLA could not be restarted; aborting queue.",
                              file=sys.stderr)
                        stop_requested["flag"] = True
                        break
                    carla_port_for_eval = handle.port

                attempt_label = f" (attempt {attempt})" if attempt > 1 else ""
                print(f"\n[hitl_run] ▶ scenario {idx}/{len(queue)}: {scenario.display}  "
                      f"(egos={scenario.ego_count}){attempt_label}")
                eval_log_path = None
                if args.quiet and scenario_dir is not None:
                    eval_log_path = scenario_dir / "eval.log"
                eval_proc = _spawn_eval(
                    route_dir=scenario.path,
                    extra_args=extra_eval_args,
                    carla_port=carla_port_for_eval,
                    dashboard_url=dashboard_url,
                    hitl_timeout_s=args.hitl_timeout_s,
                    mode=args.mode,
                    log_path=eval_log_path,
                )

                # Wait for eval to finish — either runs to completion, the
                # operator hits Ctrl-C, the operator clicks "End scenario"
                # (advance queue), or "Retry scenario" (redo this one).
                poll_iter = 0
                operator_ended = False
                operator_retry = False
                while not stop_requested["flag"]:
                    if dash_proc is not None and dash_proc.poll() is not None:
                        print(f"[hitl_run] dashboard died (rc={dash_proc.returncode}). Stopping.")
                        stop_requested["flag"] = True
                        break
                    if eval_proc.poll() is not None:
                        break
                    poll_iter += 1
                    if poll_iter % 4 == 0:
                        state = _get_dashboard(args.dashboard_host, args.dashboard_port, "/api/state")
                        if state and state.get("retry_scenario_pending"):
                            print("[hitl_run] operator clicked Retry scenario — stopping eval child for redo")
                            operator_retry = True
                            break
                        if state and state.get("end_scenario_pending"):
                            print("[hitl_run] operator clicked End scenario — stopping eval child")
                            operator_ended = True
                            break
                    time.sleep(0.5)
                rc = eval_proc.returncode if eval_proc.poll() is not None else None
                if eval_proc.poll() is None:
                    eval_proc.terminate()
                    try:
                        eval_proc.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        eval_proc.kill()
                        eval_proc.wait(timeout=5.0)
                    rc = eval_proc.returncode
                eval_proc = None

                if operator_retry and not stop_requested["flag"]:
                    print(f"[hitl_run] ↺ retrying scenario {idx}/{len(queue)} "
                          f"(prior eval rc={rc}, wiping {scenario_dir})")
                    continue  # back to top of while True — same idx, fresh attempt

                ended_by = "operator" if operator_ended else (
                    "user-interrupt" if stop_requested["flag"] else "natural"
                )
                print(f"[hitl_run] ◆ scenario {idx}/{len(queue)} ended via {ended_by} (eval rc={rc})")

                scenario_summary = _fetch_scenario_summary(
                    args.dashboard_host, args.dashboard_port, scenario, scenario_dir
                )
                per_scenario_results.append(scenario_summary)
                if experiment_dir is not None:
                    _write_experiment_summary(experiment_dir, experiment_id, queue, per_scenario_results)
                break  # advance to next scenario in queue

        if stop_requested["flag"]:
            print(f"\n[hitl_run] interrupted after {len(per_scenario_results)} of {len(queue)} scenarios.")
        else:
            print(f"\n[hitl_run] all {len(queue)} scenario(s) finished.")
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"[hitl_run] FAILED: {exc!r}", file=sys.stderr)
        return 1
    finally:
        if eval_proc is not None and eval_proc.poll() is None:
            print("[hitl_run] stopping eval...")
            eval_proc.terminate()
            try:
                eval_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                print("[hitl_run] eval did not exit, killing.")
                eval_proc.kill()
                eval_proc.wait(timeout=5.0)
        if dash_proc is not None:
            _stop_dashboard(dash_proc)
        if handle is not None:
            print("[hitl_run] stopping CARLA...")
            handle.stop()
            print("[hitl_run] CARLA stopped.")


if __name__ == "__main__":
    sys.exit(main())
