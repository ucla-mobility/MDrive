"""Glue between :mod:`tools.run_custom_eval`, :mod:`tools._carla_pool`, and
:mod:`tools._scheduler`.

This module exists to keep the surface area of ``run_custom_eval.py``
small when ``--scenario-pool`` is enabled.  It translates between the
concepts owned by ``run_custom_eval`` (``PlannerRequest``, grandchild
argv construction, env assembly, result-dir finalize path) and those
owned by the pool/scheduler.

Responsibilities
----------------
* ``extract_town_from_scenario``: discover which CARLA town a scenario
  requires, from its route XML.
* ``build_scenario_job_list``: produce a flat ``list[ScenarioJob]``
  across all planners/scenarios.
* ``spawn_grandchild_for_lease``: spawn a grandchild for one
  ``(ScenarioJob, CarlaLease)`` pair (Phase 2c will flesh this out once
  the main() integration surface is understood).
* ``finalize_result_from_run``: invoke existing
  ``write_run_status`` / ``finalize_failed_result_roots`` helpers.

Every symbol from ``run_custom_eval`` that this module needs is imported
lazily inside functions (to keep import graph shallow) or passed in
explicitly by main().
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence


log = logging.getLogger(__name__)


# Module-level registry so out-of-module watchdogs (EmergencyGuard) can
# read pool counters without coupling to the constructor call site.
# Mirrors the get_active_pool() pattern in tools/_carla_pool.py.
_active_reporter: "_PoolStatusReporter | None" = None
_active_reporter_lock = threading.Lock()


def get_active_reporter() -> "_PoolStatusReporter | None":
    """Return the currently-active _PoolStatusReporter, or None if the pool
    hasn't been constructed yet (early warmup / no scenario-pool run)."""
    with _active_reporter_lock:
        return _active_reporter


class _PoolStatusReporter:
    """Compact operator-facing status lines for the scenario pool."""

    def __init__(self, *, total_jobs: int) -> None:
        self._lock = threading.Lock()
        # Register as the active reporter so EmergencyGuard can see us.
        global _active_reporter
        with _active_reporter_lock:
            _active_reporter = self
        self._total_jobs = max(0, int(total_jobs))
        self._carla_starting = 0
        self._carla_ready = 0
        self._carla_busy = 0
        self._carla_failed_starts = 0
        self._evals_queued = self._total_jobs
        self._evals_running = 0
        self._evals_ok = 0
        self._evals_failed = 0
        # Total failure ATTEMPTS (every non-success eval finish, including
        # retried ones).  ``_evals_failed`` only counts terminal failures and
        # therefore systematically understates wasted compute and the true
        # rate at which evaluator runs are dying.  This counter exposes the
        # real total -- always >= _evals_failed.
        self._evals_failed_attempts = 0
        # Per-reason counter for total failure attempts. Lets the operator
        # see at a glance which failure mode is consuming retry budget.
        self._evals_failed_attempts_by_reason: dict[str, int] = {}
        # Per-failure latency: list of (reason, duration_s).  Lets the dashboard
        # show how quickly failures are being detected — most should be fast
        # (partial_ego_spawn, vllm_connect_error) but slow ones (sensor_failure,
        # tick_freeze) indicate watchdogs firing later than ideal.
        self._failure_latencies: list[tuple[str, float]] = []
        self._failure_latencies_max = 2000
        # Per-job in-flight tracking: job_id -> {planner, scenario, gpu, started_at}
        # Used by the periodic dashboard to show what's actually running.
        self._inflight: dict[str, dict] = {}
        # Completed runtimes for aggregate stats (planner, scenario, duration_s).
        # Bounded so it doesn't grow unbounded over very long runs.
        self._completed_runtimes: list[tuple[str, str, float]] = []
        self._completed_runtimes_max = 5000
        # Dashboard daemon thread state
        self._dashboard_started_at = time.time()
        self._dashboard_thread: threading.Thread | None = None
        self._dashboard_stop = threading.Event()
        # Stuck-killer daemon (separate from dashboard so it can't be
        # starved by slow dashboard renders).
        self._stuck_killer_thread: threading.Thread | None = None
        # Quiet mode: when ON, suppress per-event [SCENARIO_POOL] spam
        # because the dashboard already aggregates that info.  Default ON
        # whenever the dashboard is going to run.  Override via
        # CARLA_POOL_QUIET=0 to restore the verbose per-event logs.
        self._quiet_mode = (
            os.environ.get("CARLA_POOL_QUIET", "1").lower() in ("1", "true", "yes", "on")
        )

    def emit_snapshot(self, event: str, detail: str = "") -> None:
        with self._lock:
            # In quiet mode, only emit lifecycle milestones (not per-event).
            if self._quiet_mode and event not in (
                "pool_initialized", "pool_complete", "pool_interrupted",
                "pool_failed", "pool_shutdown", "pool_signal",
            ):
                return
            print(self._format_line(event=event, detail=detail), flush=True)

    def get_evals_ok(self) -> int:
        """Return the current count of successful evaluator runs.

        Read under the reporter lock so callers see a consistent value
        even if the pool is mid-update on another thread. Used by
        EmergencyGuard's no-ok-progress detector.
        """
        with self._lock:
            return int(self._evals_ok)

    def get_evals_remaining(self) -> int:
        """Return queued + currently-running eval count.

        Used by EmergencyGuard's carla_floor detector to suppress false
        positives near the end of a run: if fewer scenarios remain than
        the per-GPU CARLA floor expects, the pool will legitimately wind
        down some CARLAs and the floor would otherwise trip.
        """
        with self._lock:
            return int(self._evals_queued) + int(self._evals_running)

    def emit_failure_breakdown(self) -> None:
        """Print a breakdown of total failure attempts by decision_reason.
        Lets the operator see at a glance which failure mode dominates the
        wasted-compute share, separately from the terminal-failure count.
        Safe to call repeatedly (e.g., periodically or at shutdown)."""
        with self._lock:
            if not self._evals_failed_attempts_by_reason:
                return
            ordered = sorted(
                self._evals_failed_attempts_by_reason.items(),
                key=lambda kv: -kv[1],
            )
            parts = " ".join(f"{k}={v}" for k, v in ordered)
            print(
                f"[SCENARIO_POOL] failure_breakdown total_attempts="
                f"{self._evals_failed_attempts} terminal={self._evals_failed} "
                f"by_reason: {parts}",
                flush=True,
            )

    def _parse_inflight_log(self, log_path: str | None) -> dict:
        """
        Cheap tail-parse of a grandchild's pool log to extract live progress
        signals.  Returns a dict with whatever it could find:
          - last_frame: int  (latest sim frame the scenario reached)
          - last_tick_age_s: float  (how long since the most recent tick)
          - sim_rate_hz: float
          - ticks_total, ticks_slow: ints
          - route_completion_pct: float (if RouteCompletionTest line found,
            usually only after scenario ends -- rarely set for in-flight)
        Missing keys = signal not found.

        Reads only the last ~64 KB of the file to keep the dashboard snappy.

        ``last_tick_age_s`` priority order (newest signal wins):
          1. ``WATCHDOG_FORENSICS] heartbeat_age_s=N`` -- emitted every ~5 min
             by the leaderboard's freeze watchdog. ALWAYS reflects current
             stall duration even when ticks have stopped. This is the most
             authoritative source.
          2. SNAPSHOT's ``age_since_last_tick_s`` PLUS wall-clock time
             elapsed since the SNAPSHOT line was written. Without the
             elapsed correction, a 30s-stale snapshot would forever report
             "30s since last tick" and the stuck-killer would never fire
             on a wedged agent.
          3. SNAPSHOT's raw ``age_since_last_tick_s`` (last resort, only
             used if file mtime can't be read).

        Bug context: prior to this layered logic, scenarios stuck inside
        agent_call (no new ticks → no new SNAPSHOT) would keep reporting
        the SAME stale age forever. We observed scenarios silent for 47 min
        with the parser returning 33s as "current age", causing the 600s
        POOL_STUCK_KILLER to never fire.
        """
        result: dict = {}
        if not log_path:
            return result
        try:
            from pathlib import Path
            import re as _re
            import time as _time
            p = Path(log_path)
            if not p.exists():
                return result
            sz = p.stat().st_size
            file_mtime = p.stat().st_mtime
            now = _time.time()
            with open(p, "rb") as f:
                if sz > 65536:
                    f.seek(sz - 65536)
                tail = f.read().decode("utf-8", errors="replace")
            # CRITICAL: pool log is APPENDED to across retries of the same
            # scenario. If a previous attempt left ``WATCHDOG_FORENSICS
            # heartbeat_age_s=9154`` lines and a new attempt just started but
            # hasn't written its own data yet, naively reading the most-recent
            # heartbeat line would falsely report the fresh attempt as stuck
            # for hours. POOL_STUCK_KILLER would then murder the new attempt
            # within seconds of dispatch.
            #
            # Fix: scope the tail to ONLY data from the current attempt.
            # ``TICK_FORENSICS] INIT`` is emitted exactly once at scenario
            # start, so the most recent INIT marker delineates "this attempt's
            # data starts here". Drop everything before it.
            init_markers = list(_re.finditer(r"TICK_FORENSICS\] INIT", tail))
            if init_markers:
                tail = tail[init_markers[-1].start():]
            # Latest TICK_FORENSICS SNAPSHOT
            snaps = _re.findall(
                r"TICK_FORENSICS\] SNAPSHOT.*?last_frame=(\d+).*?last20_avg_s=([\d.]+).*?sim_rate_hz=([\d.]+).*?age_since_last_tick_s=(\S+).*?ticks_total=(\d+).*?ticks_slow=(\d+)",
                tail,
            )
            # Earlier snapshots may have ticks_total/ticks_slow before sim_rate_hz; try a more flexible fallback
            snapshot_age_at_log_time: float | None = None
            if not snaps:
                snaps = _re.findall(
                    r"TICK_FORENSICS\] SNAPSHOT.*?ticks_total=(\d+) ticks_slow=(\d+) last_frame=(\d+).*?last20_avg_s=([\d.]+).*?sim_rate_hz=([\d.]+).*?age_since_last_tick_s=(\S+)",
                    tail,
                )
                if snaps:
                    last = snaps[-1]
                    result["ticks_total"] = int(last[0])
                    result["ticks_slow"] = int(last[1])
                    result["last_frame"] = int(last[2])
                    result["sim_rate_hz"] = float(last[4])
                    age = last[5]
                    if age not in ("-1", "None"):
                        try: snapshot_age_at_log_time = float(age)
                        except: pass
            else:
                last = snaps[-1]
                result["last_frame"] = int(last[0])
                result["sim_rate_hz"] = float(last[2])
                age = last[3]
                if age not in ("-1", "None"):
                    try: snapshot_age_at_log_time = float(age)
                    except: pass
                result["ticks_total"] = int(last[4])
                result["ticks_slow"] = int(last[5])
            # Priority 1: WATCHDOG_FORENSICS heartbeat_age_s (live, updated
            # every ~5 min, accurately reflects stall duration). This MUST
            # win over stale SNAPSHOT data when both are present.
            watchdog_ages = _re.findall(
                r"WATCHDOG_FORENSICS\][^\n]*?heartbeat_age_s=([\d.]+)",
                tail,
            )
            if watchdog_ages:
                try:
                    # The most recent heartbeat line's reported age PLUS
                    # the wall-clock time elapsed since the line was written
                    # (approximated via file mtime, which is when the
                    # heartbeat line was last appended).
                    raw_age = float(watchdog_ages[-1])
                    elapsed_since_write = max(0.0, now - file_mtime)
                    result["last_tick_age_s"] = raw_age + elapsed_since_write
                except Exception:
                    pass

            # Priority 2: SNAPSHOT age + elapsed-since-write correction
            if "last_tick_age_s" not in result and snapshot_age_at_log_time is not None:
                try:
                    elapsed_since_write = max(0.0, now - file_mtime)
                    result["last_tick_age_s"] = (
                        snapshot_age_at_log_time + elapsed_since_write
                    )
                except Exception:
                    # Priority 3: raw SNAPSHOT age (fallback if mtime unreadable)
                    result["last_tick_age_s"] = snapshot_age_at_log_time
            # RouteCompletionTest (only fires after scenario end; rare for in-flight)
            rct = _re.findall(r"RouteCompletionTest\s+│?\s+\S*\s+│?\s+([\d.]+)\s*%", tail)
            if rct:
                try: result["route_completion_pct"] = float(rct[-1])
                except: pass
        except Exception:
            pass
        return result

    def emit_dashboard(self) -> None:
        """
        Periodic, structured dashboard for the operator.  Replaces the noisy
        per-event status spam with a single multi-line snapshot showing:
          - one-line counters with per-reason failure breakdown
          - aggregate runtime stats from completed jobs (avg, median, p95)
          - top-N longest-running in-flight scenarios with planner/scenario/duration

        Designed to be both human-skim-friendly and easy to grep for the
        single-line variants.
        """
        import statistics
        now = time.time()
        with self._lock:
            ok = self._evals_ok
            term_failed = self._evals_failed
            attempts_failed = self._evals_failed_attempts
            queued = self._evals_queued
            running = self._evals_running
            total = self._total_jobs
            reasons = sorted(
                self._evals_failed_attempts_by_reason.items(),
                key=lambda kv: -kv[1],
            )
            inflight_snapshot = [
                {**v, "job_id": k, "age_s": now - v["started_at"]}
                for k, v in self._inflight.items()
            ]
            completed = list(self._completed_runtimes)
            failure_lats = list(self._failure_latencies)
            uptime_s = now - self._dashboard_started_at
            carla_ready = self._carla_ready
            carla_busy = self._carla_busy

        # Compute runtime stats
        if completed:
            durs = sorted(d for _, _, d in completed)
            avg = sum(durs) / len(durs)
            median = durs[len(durs) // 2]
            p95 = durs[max(0, int(len(durs) * 0.95) - 1)]
            min_d = durs[0]
            max_d = durs[-1]
            stats_line = (
                f"runtime: n={len(durs)} avg={avg:.0f}s med={median:.0f}s "
                f"p95={p95:.0f}s min={min_d:.0f}s max={max_d:.0f}s"
            )
        else:
            stats_line = "runtime: no completed runs yet"

        # Compute success rate
        attempted = ok + term_failed
        rate = (100.0 * ok / attempted) if attempted else 0.0

        # Per-reason failure latency: how fast we detected each failure mode.
        # A bucket showing many failures with high p95 latency means the
        # watchdog for that mode is firing slowly (operator wants to know).
        from collections import defaultdict as _dd
        by_reason: dict[str, list[float]] = _dd(list)
        for r, d in failure_lats:
            by_reason[r].append(d)
        failure_lat_lines = []
        if by_reason:
            failure_lat_lines.append("failure_latency:")
            # Sort by total count descending
            for r, durs in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
                ds = sorted(durs)
                n = len(ds)
                med = ds[n // 2]
                p95 = ds[max(0, int(n * 0.95) - 1)]
                mn = ds[0]
                mx = ds[-1]
                # Tag fast/slow so the operator can spot anomalies
                tag = "FAST" if med < 60 else ("SLOW" if med > 300 else "MED")
                failure_lat_lines.append(
                    f"  {r:<40s} n={n:>3d}  med={med:>6.1f}s  p95={p95:>6.1f}s  "
                    f"min={mn:>5.1f}s  max={mx:>6.1f}s  [{tag}]"
                )

        # Augment each in-flight entry with live progress signals from its log
        for entry in inflight_snapshot:
            metrics = self._parse_inflight_log(entry.get("log_path"))
            entry.update(metrics)

        # Split running into "launching" (no first tick yet — agent loading,
        # CARLA boot, sensor warmup) vs "ticking" (world is advancing, at
        # least one TICK_FORENSICS snapshot observed).  This lets the operator
        # distinguish a slow ramp-up from a stuck dispatch.
        launching = sum(1 for e in inflight_snapshot if e.get("last_frame") is None)
        ticking = running - launching

        # Build counter line with per-reason breakdown
        if reasons:
            reason_str = ", ".join(f"{k}={v}" for k, v in reasons)
            counter_line = (
                f"running={running} (ticking={ticking} launching={launching})  "
                f"ok={ok}  failed={term_failed}/{total} "
                f"({reason_str})  attempts_failed={attempts_failed}  queued={queued}  "
                f"success_rate={rate:.1f}%"
            )
        else:
            counter_line = (
                f"running={running} (ticking={ticking} launching={launching})  "
                f"ok={ok}  failed={term_failed}/{total}  "
                f"attempts_failed={attempts_failed}  queued={queued}  "
                f"success_rate={rate:.1f}%"
            )

        # In-flight list (longest-running first, capped at 10 to keep readable)
        inflight_sorted = sorted(inflight_snapshot, key=lambda d: -d["age_s"])
        inflight_lines = []
        max_show = 10
        for entry in inflight_sorted[:max_show]:
            frame_str = (
                f"frame={entry['last_frame']:>5d}"
                if entry.get("last_frame") is not None else "frame=  ?  "
            )
            tick_age = entry.get("last_tick_age_s")
            tick_age_str = (
                f"tick_age={tick_age:>5.0f}s"
                if tick_age is not None else "tick_age=  ?  "
            )
            sim_rate = entry.get("sim_rate_hz")
            sim_rate_str = (
                f"rate={sim_rate:>5.2f}Hz"
                if sim_rate is not None else "rate=  ?  "
            )
            rct = entry.get("route_completion_pct")
            rct_str = f" route={rct:.0f}%" if rct is not None else ""
            inflight_lines.append(
                f"  gpu{entry['gpu']}  {entry['planner']:<10s} "
                f"{entry['scenario']:<45s}  age={entry['age_s']:5.0f}s  "
                f"{frame_str}  {tick_age_str}  {sim_rate_str}{rct_str}"
            )
        if len(inflight_sorted) > max_show:
            inflight_lines.append(
                f"  ... (+{len(inflight_sorted) - max_show} more in-flight)"
            )

        # Aggregate frame stats across all in-flight
        frames = [e.get("last_frame") for e in inflight_snapshot if e.get("last_frame") is not None]
        tick_ages = [e.get("last_tick_age_s") for e in inflight_snapshot if e.get("last_tick_age_s") is not None]
        rates = [e.get("sim_rate_hz") for e in inflight_snapshot if e.get("sim_rate_hz") is not None]
        if frames:
            frames_s = sorted(frames)
            n = len(frames_s)
            frame_avg = sum(frames_s) / n
            frame_med = frames_s[n // 2]
            frame_min = frames_s[0]
            frame_max = frames_s[-1]
            # Outliers: scenarios with frame > 2× median (very far ahead) OR < 0.1× median (way behind)
            outliers_high = [e for e in inflight_snapshot if e.get("last_frame") and frame_med > 0 and e["last_frame"] > frame_med * 2]
            outliers_low  = [e for e in inflight_snapshot if e.get("last_frame") is not None and frame_med > 100 and e["last_frame"] < frame_med * 0.1]
            outlier_high_str = ", ".join(f"{o['planner']}:{o['scenario']}({o['last_frame']})" for o in outliers_high[:3])
            outlier_low_str = ", ".join(f"{o['planner']}:{o['scenario']}({o['last_frame']})" for o in outliers_low[:3])
            # CARLA sim ticks at 20 Hz default → frame=N means N/20 seconds of simulated time.
            # Most route timeouts are 100-200 sim sec, so frame=2000-4000 = full route done.
            frames_line = (
                f"sim_frames(20Hz=> N*0.05s sim time): "
                f"n={n} avg={frame_avg:.0f} med={frame_med} min={frame_min} max={frame_max}"
            )
            if outliers_high:
                frames_line += f"  far-ahead: [{outlier_high_str}]"
            if outliers_low:
                frames_line += f"  way-behind: [{outlier_low_str}]"
        else:
            frames_line = "sim_frames: no data yet (parsing in-flight logs...)"
        # Stuck-detector: scenarios with last_tick_age > 60s
        stuck = [e for e in inflight_snapshot if (e.get("last_tick_age_s") or 0) > 60]
        if tick_ages:
            avg_age = sum(tick_ages) / len(tick_ages)
            max_age = max(tick_ages)
            ticks_line = (
                f"tick_age: avg={avg_age:.0f}s max={max_age:.0f}s  stuck(>60s)={len(stuck)}/{len(inflight_snapshot)}"
            )
            if stuck:
                stuck_str = ", ".join(
                    f"{e['planner']}:{e['scenario']}({(e.get('last_tick_age_s') or 0):.0f}s)"
                    for e in sorted(stuck, key=lambda d: -(d.get('last_tick_age_s') or 0))[:3]
                )
                ticks_line += f"  worst: [{stuck_str}]"
        else:
            ticks_line = "tick_age: no data yet"

        if rates:
            rates_s = sorted(rates)
            rate_avg = sum(rates_s) / len(rates_s)
            rate_med = rates_s[len(rates_s) // 2]
            rate_min = rates_s[0]
            rate_max = rates_s[-1]
            rates_line = (
                f"sim_rate: avg={rate_avg:.2f}Hz med={rate_med:.2f}Hz min={rate_min:.2f}Hz max={rate_max:.2f}Hz"
            )
        else:
            rates_line = "sim_rate: no data yet"

        # Per-route aggregate runtimes (top 5 slowest by avg).  Useful for
        # spotting consistently-slow scenarios across planners.
        per_route: dict = {}
        for planner, scen, dur in completed:
            per_route.setdefault(scen, []).append((planner, dur))
        slow_routes = []
        for scen, entries in per_route.items():
            durs = [d for _, d in entries]
            avg = sum(durs) / len(durs)
            slow_routes.append((scen, len(entries), avg, max(durs)))
        slow_routes.sort(key=lambda x: -x[2])  # by avg desc
        slow_routes_line = ""
        if slow_routes:
            top = slow_routes[:5]
            parts = [f"{scen}(n={n},avg={avg:.0f}s)" for scen, n, avg, _ in top]
            slow_routes_line = "slow_routes(top5_by_avg): " + " ".join(parts)

        # Aggregate route_completion stats from BOTH the per-scenario logs
        # (in-flight final-result lines) and per-job POOL DECISION snapshots.
        # We parse RouteCompletionTest values out of every recent log tail.
        route_pcts = []
        completed_route_progress = []  # (scen, route_done, route_total) from finished jobs
        for entry in inflight_snapshot:
            rct = entry.get("route_completion_pct")
            if rct is not None:
                route_pcts.append(rct)
        # Walk recent log tails for completed jobs to gather route_progress
        from pathlib import Path as _P
        try:
            log_dir = _P("/data2/marco/CoLMDriver/results/results_driving_custom/baseline/_planner_logs")
            import re as _re
            import time as _time
            cutoff = _time.time() - 3600  # last hour
            recent = sorted(log_dir.glob("*.log"), key=lambda f: -f.stat().st_mtime)[:80]
            for log in recent:
                if log.stat().st_mtime < cutoff: continue
                try:
                    sz = log.stat().st_size
                    with open(log, "rb") as f:
                        if sz > 32768: f.seek(sz - 32768)
                        tail = f.read().decode("utf-8", errors="replace")
                except Exception:
                    continue
                # Parse RouteCompletionTest (e.g. "│ FAILURE │ 27.5 % │")
                for m in _re.finditer(r"RouteCompletionTest\s+│?\s+\S*\s+│?\s+([\d.]+)\s*%", tail):
                    try: route_pcts.append(float(m.group(1)))
                    except: pass
        except Exception:
            pass
        if route_pcts:
            rp_s = sorted(route_pcts)
            n = len(rp_s)
            mean = sum(rp_s) / n
            med = rp_s[n // 2]
            zero_pct_count = sum(1 for r in rp_s if r < 1.0)
            full_pct_count = sum(1 for r in rp_s if r >= 99.0)
            route_completion_line = (
                f"route_completion: n={n} mean={mean:.1f}% med={med:.1f}% "
                f"min={rp_s[0]:.1f}% max={rp_s[-1]:.1f}%  "
                f"zero({zero_pct_count}) full({full_pct_count})"
            )
        else:
            route_completion_line = "route_completion: no scored routes yet"

        # Format: timestamp + boxed sections
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        sep = "═" * 100

        lines = [
            "",
            sep,
            f"║ POOL DASHBOARD  {ts}  uptime={uptime_s/60:.1f}min  CARLAs(ready/busy)={carla_ready}/{carla_busy}",
            sep,
            f"║ {counter_line}",
            f"║ {stats_line}",
            f"║ {frames_line}",
            f"║ {ticks_line}",
            f"║ {rates_line}",
            f"║ {route_completion_line}",
        ]
        if slow_routes_line:
            lines.append(f"║ {slow_routes_line}")
        for ln in failure_lat_lines:
            lines.append(f"║ {ln}")
        if inflight_lines:
            lines.append(
                f"║ in-flight ({len(inflight_sorted)}, of which "
                f"ticking={ticking} launching={launching}, longest first):"
            )
            for ln in inflight_lines:
                lines.append(f"║ {ln}")
            # If launching jobs were truncated by the max_show cap, list them
            # explicitly so the operator can see what's still booting.
            shown_jids = {entry.get("job_id") for entry in inflight_sorted[:max_show]}
            launching_not_shown = [
                e for e in inflight_snapshot
                if e.get("last_frame") is None and e.get("job_id") not in shown_jids
            ]
            if launching_not_shown:
                lines.append(f"║   launching (not in top-{max_show}):")
                for e in launching_not_shown[:20]:
                    lines.append(
                        f"║     gpu{e['gpu']}  {e['planner']:<12s} "
                        f"{e['scenario']:<45s}  age={e['age_s']:5.0f}s"
                    )
                if len(launching_not_shown) > 20:
                    lines.append(
                        f"║     ... (+{len(launching_not_shown)-20} more launching)"
                    )
        lines.append(sep)
        lines.append("")
        # One print to keep it atomic
        print("\n".join(lines), flush=True)

    def start_dashboard(self, *, interval_s: float = 60.0) -> None:
        """Start the periodic dashboard daemon thread.  Safe to call once."""
        if self._dashboard_thread is not None:
            return
        self._dashboard_thread = threading.Thread(
            target=self._dashboard_loop,
            args=(float(interval_s),),
            daemon=True,
            name="pool_status_dashboard",
        )
        self._dashboard_thread.start()

    def start_stuck_killer(self, *, interval_s: float = 60.0) -> None:
        """Start the dedicated stuck-killer daemon.  Independent of the
        dashboard so a slow dashboard render can't delay kills.  Safe to
        call once.  Disable entirely via CARLA_POOL_STUCK_KILL_DISABLE=1.
        """
        if self._stuck_killer_thread is not None:
            return
        if os.environ.get("CARLA_POOL_STUCK_KILL_DISABLE", "0").lower() in ("1", "true", "yes"):
            print("[POOL_STUCK_KILLER] disabled via CARLA_POOL_STUCK_KILL_DISABLE", flush=True)
            return
        self._stuck_killer_thread = threading.Thread(
            target=self._stuck_killer_loop,
            args=(float(interval_s),),
            daemon=True,
            name="pool_stuck_killer",
        )
        self._stuck_killer_thread.start()
        print(
            f"[POOL_STUCK_KILLER] daemon started, interval={interval_s:.0f}s, "
            f"threshold={float(os.environ.get('CARLA_POOL_STUCK_KILL_S', '600')):.0f}s",
            flush=True,
        )

    def stop_dashboard(self) -> None:
        self._dashboard_stop.set()

    def _dashboard_loop(self, interval_s: float) -> None:
        # Brief delay so the first dashboard isn't empty.
        if self._dashboard_stop.wait(min(interval_s, 10.0)):
            return
        while not self._dashboard_stop.wait(interval_s):
            try:
                self.emit_dashboard()
            except Exception:
                pass
            # NOTE: stuck-killer used to live here, coupled to the dashboard
            # interval.  That coupling meant a slow dashboard render (large
            # in-flight count parsing) could delay the killer indefinitely,
            # which is exactly what we observed in the run that piled 23
            # stuck CARLAs over 8 hours.  The killer now runs on its own
            # daemon thread (start_stuck_killer) so it can't be starved.

    def _stuck_killer_loop(self, interval_s: float) -> None:
        """Dedicated daemon thread for the stuck-scenario killer.  Decoupled
        from the dashboard so a slow dashboard render can't delay this."""
        # Brief grace period at startup so first-wave scenarios have time to
        # boot CARLA + load models before being judged for tick freshness.
        if self._dashboard_stop.wait(min(interval_s, 30.0)):
            return
        ticks = 0
        while not self._dashboard_stop.wait(interval_s):
            ticks += 1
            try:
                self._maybe_kill_stuck_scenarios()
            except Exception:
                import traceback
                print(
                    f"[POOL_STUCK_KILLER_HEARTBEAT] tick={ticks} "
                    f"ERROR: {traceback.format_exc(limit=2)}",
                    flush=True,
                )
                continue
            # Every 5 ticks (= ~5 min at 60s interval), emit a heartbeat so
            # the operator can verify the daemon is alive.
            if ticks % 5 == 0:
                try:
                    with self._lock:
                        n_inflight = len(self._inflight)
                    print(
                        f"[POOL_STUCK_KILLER_HEARTBEAT] tick={ticks} "
                        f"in_flight={n_inflight} threshold_s="
                        f"{float(os.environ.get('CARLA_POOL_STUCK_KILL_S', '600')):.0f}",
                        flush=True,
                    )
                except Exception:
                    pass

    def _maybe_kill_stuck_scenarios(self) -> None:
        """Detect and kill in-flight scenarios that have produced ZERO new
        sim frames for the configured stuck threshold.  Default: 30 min.

        Different from the 24hr heartbeat watchdog -- this fires much sooner
        for scenarios that demonstrably ARE truly frozen (last_tick_age >
        stuck_threshold).  Slow-but-progressing scenarios (any tick within
        the threshold) are NEVER killed by this.

        Disabled with CARLA_POOL_STUCK_KILL_DISABLE=1 or threshold=0.
        Threshold configurable via CARLA_POOL_STUCK_KILL_S (default 1800s).
        """
        if os.environ.get("CARLA_POOL_STUCK_KILL_DISABLE", "0").lower() in ("1", "true", "yes"):
            return
        try:
            # Default: 600s (10 min).  Justified by measured tick distribution
            # across 2340 SNAPSHOT samples in this run:
            #   median=3.85s  p90=75s  p95=94s  p99=545s  max=557s
            # The p99 / max are dominated by legitimate "very slow ticks"
            # (CARLA physics pauses, slow agent inference under contention).
            # 600s = 1.1x the worst observed legitimate slow tick — anything
            # beyond this is genuinely frozen, not just slow.  Lower thresholds
            # (300s) would false-positive on ~1.2% of normal slow-tick events.
            # Was 1800s (30 min) — too patient given the data.
            stuck_threshold_s = float(os.environ.get("CARLA_POOL_STUCK_KILL_S", "600"))
        except (TypeError, ValueError):
            stuck_threshold_s = 600.0
        if stuck_threshold_s <= 0:
            return

        with self._lock:
            inflight_snapshot = [
                {**v, "job_id": k} for k, v in self._inflight.items()
            ]
        # Augment with parsed metrics
        candidates = []
        for entry in inflight_snapshot:
            metrics = self._parse_inflight_log(entry.get("log_path"))
            age = metrics.get("last_tick_age_s")
            if age is not None and age >= stuck_threshold_s:
                candidates.append({**entry, **metrics})
        if not candidates:
            return

        for entry in candidates:
            world_port = entry.get("port", 0)
            if not world_port:
                continue
            killed = self._kill_grandchild_by_world_port(world_port)
            print(
                f"[POOL_STUCK_KILLER] killed_pid={killed} "
                f"planner={entry.get('planner')} scenario={entry.get('scenario')} "
                f"port={world_port} last_frame={entry.get('last_frame')} "
                f"last_tick_age_s={entry.get('last_tick_age_s'):.0f}",
                flush=True,
            )

    @staticmethod
    def _kill_grandchild_by_world_port(world_port: int) -> int | None:
        """Find the grandchild leaderboard_evaluator_parameter process bound
        to ``world_port`` and signal it. Returns the targeted PID or None
        if not found.

        Two-stage signal escalation:

          1. Send **SIGINT** to the process group. The leaderboard registers
             a SIGINT handler at ``leaderboard_evaluator_parameter.py:253``
             which forwards to ``ScenarioManager.signal_handler`` -- this
             sets ``self._running = False`` so the main loop exits at its
             next iteration. Once the loop exits, the evaluator runs its
             normal termination path including ``_register_statistics()``
             which writes per-ego ``results.json`` with the criteria state
             at kill time. The recorded status reflects what the in-process
             criteria observed (e.g. ``"Failed - Agent timed out"`` for a
             stuck ego) -- NOT a fake "success".

          2. After ``CARLA_POOL_STUCK_KILL_GRACE_S`` seconds (default 60),
             if the process is still alive (signal was queued but never
             delivered because the main thread is wedged in C-extension
             code such as a futex / RPC stall), escalate to **SIGKILL**.

        Why not SIGTERM as before? SIGTERM has no handler in the leaderboard
        and immediately kills the process. ``_register_statistics()`` never
        runs, per-ego ``results.json`` is empty, and the wrapper's
        classifier marks the run as a hard infrastructure failure (``rc=-15``
        / ``rc=-9``) instead of recording the actual planner-stall outcome.

        Why not block waiting for clean exit? The escalation runs in a
        daemon thread so the stuck-killer can move on to other candidates
        without serialising 60s per kill.
        """
        import subprocess as _sp
        import signal as _sig
        try:
            out = _sp.check_output(
                ["ps", "-eo", "pid,cmd"], text=True, errors="replace"
            )
        except Exception:
            return None
        target = f"--port={world_port}"
        target_alt = f"--port {world_port}"
        target_pid = None
        target_pgid = None
        for line in out.splitlines():
            if "leaderboard_evaluator_parameter" not in line:
                continue
            if target not in line and target_alt not in line:
                continue
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            try:
                target_pid = int(parts[0])
            except ValueError:
                continue
            try:
                target_pgid = os.getpgid(target_pid)
            except (OSError, ProcessLookupError):
                target_pgid = None
            break

        if target_pid is None:
            return None

        try:
            grace_s = float(os.environ.get("CARLA_POOL_STUCK_KILL_GRACE_S", "60"))
        except (TypeError, ValueError):
            grace_s = 60.0
        grace_s = max(5.0, grace_s)  # floor: 5s, otherwise pointless

        # Stage 1: SIGINT (graceful) -- handled by leaderboard, triggers
        # _register_statistics on its way out.
        sigint_sent = False
        if target_pgid is not None:
            try:
                os.killpg(target_pgid, _sig.SIGINT)
                sigint_sent = True
            except (OSError, ProcessLookupError):
                pass
        if not sigint_sent:
            try:
                os.kill(target_pid, _sig.SIGINT)
                sigint_sent = True
            except (OSError, ProcessLookupError):
                return None
        print(
            f"[POOL_STUCK_KILLER] sent SIGINT pid={target_pid} "
            f"pgid={target_pgid} -- waiting {grace_s:.0f}s for graceful exit "
            f"with results.json before SIGKILL escalation",
            flush=True,
        )

        # Stage 2: schedule SIGKILL escalation in a daemon thread so the
        # caller (stuck-killer loop) doesn't block.
        def _escalate(pid: int, pgid: int | None, deadline_s: float) -> None:
            time.sleep(deadline_s)
            # Check if still alive
            try:
                os.kill(pid, 0)  # signal 0 = check only
                still_alive = True
            except (OSError, ProcessLookupError):
                still_alive = False
            if not still_alive:
                print(
                    f"[POOL_STUCK_KILLER] pid={pid} exited cleanly within "
                    f"{deadline_s:.0f}s grace; results.json should be on disk",
                    flush=True,
                )
                return
            # Still alive after grace period; SIGINT didn't take.
            # Escalate to SIGKILL on the process group.
            print(
                f"[POOL_STUCK_KILLER] pid={pid} still alive after "
                f"{deadline_s:.0f}s SIGINT grace; escalating to SIGKILL "
                f"(process likely wedged in C-extension code -- results.json "
                f"may NOT be written for this scenario, wrapper will mark "
                f"as hard failure)",
                flush=True,
            )
            try:
                if pgid is not None:
                    os.killpg(pgid, _sig.SIGKILL)
                else:
                    os.kill(pid, _sig.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

        threading.Thread(
            target=_escalate,
            args=(target_pid, target_pgid, grace_s),
            name=f"stuck_kill_escalate_{target_pid}",
            daemon=True,
        ).start()
        return target_pid

    def on_carla_event(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            detail = self._format_carla_detail(event, payload)
            if event == "carla_start_begin":
                self._carla_starting += 1
            elif event == "carla_start_ready":
                self._carla_starting = max(0, self._carla_starting - 1)
                self._carla_ready += 1
            elif event == "carla_start_failed":
                self._carla_starting = max(0, self._carla_starting - 1)
                self._carla_failed_starts += 1
            elif event == "carla_lease_acquired":
                self._carla_busy += 1
            elif event == "carla_lease_released":
                self._carla_busy = max(0, self._carla_busy - 1)
            elif event == "carla_stopped":
                self._carla_ready = max(0, self._carla_ready - 1)
            else:
                return
            if not self._quiet_mode:
                print(self._format_line(event=event, detail=detail), flush=True)

    def on_eval_start(self, *, job: Any, lease: Any, log_path=None) -> None:
        with self._lock:
            self._evals_queued = max(0, self._evals_queued - 1)
            self._evals_running += 1
            self._inflight[str(getattr(job, "job_id", "?"))] = {
                "planner": str(getattr(job, "planner_name", "?")),
                "scenario": str(getattr(job, "scenario_name", "?")),
                "gpu": int(getattr(lease, "gpu_id", -1)),
                "port": int(getattr(lease, "world_port", 0)),
                "started_at": time.time(),
                "log_path": str(log_path) if log_path else None,
            }
            if not self._quiet_mode:
                detail = (
                    f"planner={job.planner_name} scenario={job.scenario_name} "
                    f"gpu={lease.gpu_id} port={lease.world_port}"
                )
                print(self._format_line(event="eval_start", detail=detail), flush=True)

    def on_eval_finish(self, *, result: Any) -> None:
        with self._lock:
            self._evals_running = max(0, self._evals_running - 1)
            status_value = str(result.status.value)
            # Pop in-flight entry and record runtime
            jid = str(getattr(result.job, "job_id", "?"))
            popped = self._inflight.pop(jid, None)
            try:
                _dur = float(getattr(result, "duration_s", 0.0) or 0.0)
                _planner = str(getattr(result.job, "planner_name", "?"))
                _scenario = str(getattr(result.job, "scenario_name", "?"))
                self._completed_runtimes.append((_planner, _scenario, _dur))
                if len(self._completed_runtimes) > self._completed_runtimes_max:
                    self._completed_runtimes = self._completed_runtimes[-self._completed_runtimes_max:]
            except Exception:
                pass
            if status_value == "success":
                self._evals_ok += 1
                event = "eval_done"
            elif status_value == "retry_queued":
                # Retry will re-enter the queue; put it back on ``queued``.
                self._evals_queued += 1
                event = "eval_retry"
                # Count this as a failure attempt (it failed, just gets retried).
                self._evals_failed_attempts += 1
                _reason = str(getattr(result, "decision_reason", "") or "unknown")
                self._evals_failed_attempts_by_reason[_reason] = (
                    self._evals_failed_attempts_by_reason.get(_reason, 0) + 1
                )
                self._failure_latencies.append((_reason, float(_dur)))
            else:
                self._evals_failed += 1
                self._evals_failed_attempts += 1
                _reason = str(getattr(result, "decision_reason", "") or "unknown")
                self._evals_failed_attempts_by_reason[_reason] = (
                    self._evals_failed_attempts_by_reason.get(_reason, 0) + 1
                )
                self._failure_latencies.append((_reason, float(_dur)))
                event = "eval_done"
            if len(self._failure_latencies) > self._failure_latencies_max:
                self._failure_latencies = self._failure_latencies[-self._failure_latencies_max:]
            detail = (
                f"planner={result.job.planner_name} scenario={result.job.scenario_name} "
                f"status={status_value} gpu={result.lease_gpu_id} "
                f"dur={result.duration_s:.1f}s"
            )
            if status_value != "success":
                summary = self._summarize_failure_tail(
                    str(getattr(result, "stderr_tail", "") or "")
                )
                if summary:
                    detail += f" tail={summary[:200]}"
                if status_value == "retry_queued":
                    detail += (
                        f" attempt={int(getattr(result.job, 'retry_attempt', 0)) + 1}"
                        f"/{int(getattr(result.job, 'retry_budget', 0))}"
                    )
            # In quiet mode, only emit eval_done lines for FAILED jobs (so
            # operator still sees failure detail at the moment it happens),
            # and suppress eval_done for successes + retry_queued (those
            # are aggregated in the dashboard).
            if not self._quiet_mode or status_value not in ("success", "retry_queued"):
                print(self._format_line(event=event, detail=detail), flush=True)

    def _format_line(self, *, event: str, detail: str) -> str:
        # Per-reason breakdown after the main counters, e.g.:
        # failed=15/373 (catchall_terminal=10, partial_ego_spawn=3, alignment_failed=2)
        if self._evals_failed_attempts_by_reason:
            ordered = sorted(
                self._evals_failed_attempts_by_reason.items(),
                key=lambda kv: -kv[1],
            )
            reason_str = ", ".join(f"{k}={v}" for k, v in ordered)
            failed_breakdown = f" ({reason_str})"
        else:
            failed_breakdown = ""
        return (
            "[SCENARIO_POOL] "
            f"CARLAs starting={self._carla_starting} rpc_ready={self._carla_ready} "
            f"busy={self._carla_busy} failed_starts={self._carla_failed_starts} | "
            f"evals queued={self._evals_queued} running={self._evals_running} "
            f"ok={self._evals_ok} failed={self._evals_failed}/{self._total_jobs}"
            f"{failed_breakdown} "
            f"attempts_failed={self._evals_failed_attempts} | "
            f"event={event}"
            + (f" {detail}" if detail else "")
        )

    @staticmethod
    def _format_carla_detail(event: str, payload: dict[str, Any]) -> str:
        gpu = payload.get("gpu_id")
        world_port = payload.get("world_port")
        tm_port = payload.get("tm_port")
        duration_s = payload.get("duration_s")
        parts: list[str] = []
        if gpu is not None:
            parts.append(f"gpu={gpu}")
        if world_port is not None:
            parts.append(f"port={world_port}")
        if tm_port is not None:
            parts.append(f"tm_port={tm_port}")
        if isinstance(duration_s, (int, float)):
            parts.append(f"dur={float(duration_s):.1f}s")
        reason = str(payload.get("reason") or "").strip()
        if reason and event == "carla_start_failed":
            parts.append(f"reason={reason}")
        status = str(payload.get("status") or "").strip()
        recycled = payload.get("recycled")
        if event == "carla_lease_released" and status:
            parts.append(f"status={status}")
        if event == "carla_lease_released" and recycled is not None:
            parts.append(f"recycled={bool(recycled)}")
        return " ".join(parts)

    @staticmethod
    def _summarize_failure_tail(stderr_tail: str) -> str:
        lines = [line.strip() for line in str(stderr_tail or "").splitlines() if line.strip()]
        if not lines:
            return ""

        skip_prefixes = (
            "ERROR conda.cli.main_run:execute(",
        )
        preferred_markers = (
            "RuntimeError:",
            "ValueError:",
            "TypeError:",
            "AssertionError:",
            "ImportError:",
            "ModuleNotFoundError:",
            "Traceback",
            "error:",
            "Exception:",
        )

        for line in reversed(lines):
            if any(marker in line for marker in preferred_markers):
                return line
        for line in reversed(lines):
            if not any(line.startswith(prefix) for prefix in skip_prefixes):
                return line
        return lines[-1]


# ----------------------------------------------------------------------
# Town extraction
# ----------------------------------------------------------------------


def extract_town_from_scenario(
    scenario_dir: Path,
    *,
    fallback_town: str = "Town01",
) -> str:
    """Parse the first ego-role route XML under ``scenario_dir`` and return
    its ``town`` attribute.

    Returns ``fallback_town`` if no ego route is found or the XML is
    unreadable.  Does NOT raise -- scheduling must continue even if a
    scenario has an unusual layout.
    """
    from tools.setup_scenario_from_zip import parse_route_metadata

    if not scenario_dir.is_dir():
        log.warning("scenario_dir does not exist: %s", scenario_dir)
        return fallback_town

    for xml_path in sorted(scenario_dir.rglob("*.xml")):
        if "actors" in xml_path.parts:
            continue
        if "_REPLAY" in xml_path.stem:
            # Log-replay twin files share the same town as the primary.
            continue
        try:
            _route_id, town, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role != "ego":
            continue
        if town:
            return town

    # No ego route found; fall back.
    return fallback_town


# ----------------------------------------------------------------------
# Pre-flight scenario-data validation (P1)
# ----------------------------------------------------------------------


_EGO_SPAWN_COLLISION_MIN_DIST_M = 3.0
_EGO_MIN_WAYPOINTS = 2


def _parse_ego_first_waypoint(xml_path: Path) -> tuple[str, float, float] | None:
    """Return ``(route_id_or_filename, x, y)`` for the first <waypoint>
    element of an ego-role route XML, or None if the file is unreadable
    or lacks a usable waypoint.
    """
    import xml.etree.ElementTree as _ET
    try:
        tree = _ET.parse(xml_path)
    except Exception:
        return None
    root = tree.getroot()
    for route in root.findall(".//route"):
        role = str(route.get("role", "")).strip().lower()
        if role and role != "ego":
            continue
        wp = route.find("waypoint")
        if wp is None:
            continue
        try:
            x = float(wp.get("x"))
            y = float(wp.get("y"))
        except (TypeError, ValueError):
            continue
        ident = str(route.get("id") or xml_path.name)
        return (ident, x, y)
    return None


def _count_ego_waypoints(xml_path: Path) -> int:
    import xml.etree.ElementTree as _ET
    try:
        tree = _ET.parse(xml_path)
    except Exception:
        return 0
    root = tree.getroot()
    max_wps = 0
    for route in root.findall(".//route"):
        role = str(route.get("role", "")).strip().lower()
        if role and role != "ego":
            continue
        count = len(route.findall("waypoint"))
        if count > max_wps:
            max_wps = count
    return max_wps


def validate_scenario_data(
    scenario_dir: Path,
    *,
    ego_min_dist_m: float = _EGO_SPAWN_COLLISION_MIN_DIST_M,
    min_waypoints: int = _EGO_MIN_WAYPOINTS,
) -> tuple[bool, str]:
    """Static pre-flight check on a scenario directory's ego XMLs.

    Returns ``(ok, reason)``.  ``ok=False`` means the scenario will
    deterministically fail in CARLA (duplicate ego spawn, degenerate
    route, missing XMLs) and should NEVER be queued -- queueing it just
    wastes 200-500s of CARLA time to reach the same conclusion
    downstream.

    The checks are intentionally conservative: each one is a known
    deterministic failure pattern we've observed in the baseline run.
    New patterns can be added as we learn them, but false positives
    are costly (a real scenario dropped for no reason) so we only
    reject on high-confidence signals.
    """
    if not scenario_dir.is_dir():
        return (False, f"scenario_dir_not_found:{scenario_dir}")

    ego_xmls: list[Path] = []
    for xml_path in sorted(scenario_dir.rglob("*.xml")):
        # Skip actor / replay / static spec files.
        parts = xml_path.parts
        if "actors" in parts:
            continue
        if "_REPLAY" in xml_path.stem:
            continue
        if not any(
            marker in xml_path.stem.lower()
            for marker in ("ego", "vehicle_1_", "vehicle_2_", "vehicle_3_",
                           "vehicle_4_", "vehicle_5_", "vehicle_6_",
                           "vehicle_7_", "vehicle_8_")
        ) and "placement" not in xml_path.stem.lower():
            continue
        ego_xmls.append(xml_path)

    # It's fine for a scenario to have ego XMLs under the top level
    # with arbitrary names (e.g. "07_placement"); rather than rely on
    # naming, also parse and check whether each claimed ego route has
    # the "role='ego'" attribute.
    ego_positions: list[tuple[Path, str, float, float]] = []
    for xml_path in ego_xmls:
        wp = _parse_ego_first_waypoint(xml_path)
        if wp is None:
            continue
        wp_count = _count_ego_waypoints(xml_path)
        if wp_count < min_waypoints:
            return (
                False,
                f"ego_route_too_short:{xml_path.name}:waypoints={wp_count}",
            )
        route_id, x, y = wp
        ego_positions.append((xml_path, route_id, x, y))

    # Check duplicate spawn positions between ego XMLs.
    for i in range(len(ego_positions)):
        for j in range(i + 1, len(ego_positions)):
            _, idi, xi, yi = ego_positions[i]
            _, idj, xj, yj = ego_positions[j]
            dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if dist < ego_min_dist_m:
                return (
                    False,
                    f"duplicate_ego_spawn:{idi}_vs_{idj}:dist={dist:.2f}m",
                )
    return (True, "ok")


def extract_towns_per_scenario(
    scenario_entries: Iterable[tuple[Path, str]],
    *,
    fallback_town: str = "Town01",
) -> dict[str, str]:
    """Return ``{scenario_name: town}`` for a sequence of
    ``(scenario_dir, scenario_name)`` entries.

    ``scenario_entries`` is the shape returned by
    :func:`tools.run_custom_eval.detect_scenario_subfolders`.
    """
    out: dict[str, str] = {}
    for scenario_dir, scenario_name in scenario_entries:
        out[scenario_name] = extract_town_from_scenario(
            scenario_dir, fallback_town=fallback_town
        )
    return out


def filter_pending_pool_scenario_entries(
    *,
    planner_requests: list,  # list[PlannerRequest]
    scenario_entries: list[tuple[Path, str]],
    repo_root: Path,
    results_tag_base: str,
    args,
) -> tuple[dict[str, list[tuple[Path, str]]], dict[str, int], dict[str, int]]:
    """Return per-planner scenario entries that still need evaluation.

    Pool mode should never enqueue a planner/scenario pair when completed
    result roots already exist for that exact destination.  This helper
    applies the same result-root planning logic the grandchild will use and
    drops completed work before the scheduler queue is built.
    """
    from tools.run_custom_eval import (
        combine_results_subdir,
        detect_ego_routes,
        filter_scenario_entries_for_shard,
        planned_result_roots,
        result_roots_have_completed_outcomes,
    )

    pending_entries_by_planner: dict[str, list[tuple[Path, str]]] = {}
    candidate_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    ego_count_cache: dict[tuple[str, bool], int] = {}
    # Cache pre-flight validation per scenario_dir; it's the same for
    # all planners so run it once.
    validation_cache: dict[str, tuple[bool, str]] = {}

    for planner in planner_requests:
        planner_run_id = str(planner.run_id or planner.name)
        replay_mode = str(planner.name or "").strip().lower() == "log-replay"
        planner_entries = list(
            filter_scenario_entries_for_shard(
                scenario_entries,
                getattr(planner, "scenario_shard", None),
            )
        )
        candidate_counts[planner_run_id] = len(planner_entries)
        skipped_counts[planner_run_id] = 0

        pending_entries: list[tuple[Path, str]] = []
        for scenario_dir, scenario_name in planner_entries:
            scenario_dir = Path(scenario_dir)
            scenario_name = str(scenario_name or scenario_dir.name)
            ego_cache_key = (str(scenario_dir), replay_mode)
            ego_count = ego_count_cache.get(ego_cache_key)
            if ego_count is None:
                ego_count = max(
                    1,
                    detect_ego_routes(
                        scenario_dir,
                        replay_mode=replay_mode,
                    ),
                )
                ego_count_cache[ego_cache_key] = ego_count

            scenario_results_subdir = combine_results_subdir(
                getattr(args, "results_subdir", None),
                scenario_name,
            )
            effective_results_tag = (
                f"{results_tag_base}/{planner_run_id}/{scenario_results_subdir}"
            )
            result_roots = planned_result_roots(
                repo_root,
                effective_results_tag,
                args,
            )
            if result_roots_have_completed_outcomes(
                result_roots,
                ego_count=ego_count,
            ):
                skipped_counts[planner_run_id] += 1
                log.info(
                    "scenario_pool: skipping completed planner=%s scenario=%s",
                    planner_run_id,
                    scenario_name,
                )
                continue

            # Pre-flight data validation: reject scenarios with duplicate
            # ego spawn coordinates, 1-waypoint ego routes, etc.  These
            # will deterministically fail in CARLA; running them wastes
            # 200-500s of compute to reach the same conclusion.  Gated
            # on --no-pool-preflight-validate for operators who want to
            # force-include everything.
            if not bool(getattr(args, "pool_no_preflight_validate", False)):
                cache_key = str(scenario_dir)
                validation = validation_cache.get(cache_key)
                if validation is None:
                    validation = validate_scenario_data(scenario_dir)
                    validation_cache[cache_key] = validation
                ok, reason = validation
                if not ok:
                    skipped_counts[planner_run_id] += 1
                    log.warning(
                        "scenario_pool: PREFLIGHT_REJECT planner=%s scenario=%s reason=%s",
                        planner_run_id,
                        scenario_name,
                        reason,
                    )
                    continue

            pending_entries.append((scenario_dir, scenario_name))

        pending_entries_by_planner[planner_run_id] = pending_entries

    return pending_entries_by_planner, candidate_counts, skipped_counts


# ----------------------------------------------------------------------
# Job construction
# ----------------------------------------------------------------------


def build_scenario_job_list(
    *,
    planner_requests: list,                # list[PlannerRequest]
    scenario_entries: list,                # list[tuple[Path, str]] or [(Path, None)] for single
    scenario_entries_by_planner: dict[str, list] | None = None,
    results_root: Path,
    results_tag_base: str,
    retry_budget: int,
    vram_estimator,                        # PlannerVramEstimator
    fallback_town: str = "Town01",
) -> list:
    """Produce ``list[ScenarioJob]`` across all planners/scenarios.

    For each (planner, scenario) pair:
      * VRAM estimate comes from ``vram_estimator.get(planner_name)``
        (already populated from the ``--planner-vram-estimate-cache`` file
        and any ``--planner-vram-estimate-mib`` overrides).
      * Town is parsed from the scenario's ego route XML.
      * ``results_tag`` is ``"{results_tag_base}/{planner.run_id}"`` --
        the same scheme today's planner-child layer uses.
    """
    from tools._scheduler import ScenarioJob

    jobs: list = []
    town_cache: dict[str, str] = {}

    for planner in planner_requests:
        planner_name: str = planner.name
        planner_run_id: str = planner.run_id or planner.name
        planner_entries = list(
            (scenario_entries_by_planner or {}).get(planner_run_id, scenario_entries)
        )
        try:
            vram_estimate_mib = int(vram_estimator.estimate(planner_name))
        except Exception:
            vram_estimate_mib = 0  # scheduler will treat as "unknown, skip"
        for idx, entry in enumerate(planner_entries):
            if isinstance(entry, tuple) and len(entry) == 2:
                scenario_dir, scenario_name = entry
            else:
                scenario_dir = entry  # fallback: entry is a Path
                scenario_name = Path(scenario_dir).name
            if scenario_name not in town_cache:
                town_cache[scenario_name] = extract_town_from_scenario(
                    Path(scenario_dir), fallback_town=fallback_town
                )
            jobs.append(
                ScenarioJob(
                    job_id=f"{planner_run_id}:{scenario_name}:{idx:04d}",
                    planner_run_id=planner_run_id,
                    planner_name=planner_name,
                    scenario_name=str(scenario_name),
                    scenario_dir=Path(scenario_dir),
                    results_tag=f"{results_tag_base}/{planner_run_id}",
                    town=town_cache[scenario_name],
                    vram_estimate_mib=vram_estimate_mib,
                    retry_budget=int(retry_budget),
                    spawn_kwargs={
                        "conda_env": planner.conda_env,
                    },
                )
            )
    return jobs


# ----------------------------------------------------------------------
# Grandchild spawn
# ----------------------------------------------------------------------


def build_grandchild_pool_env(
    *,
    parent_env: dict[str, str],
    lease,                                 # CarlaLease
) -> dict[str, str]:
    """Inject the CARLA_POOL_* handshake env vars into ``parent_env``.

    The grandchild evaluator inspects ``CARLA_POOL_MANAGED=1`` and, when
    set, SKIPS its own ``CarlaProcessManager`` startup and instead
    connects directly to ``(CARLA_POOL_HOST, CARLA_POOL_WORLD_PORT)``.

    Other vars are informational (used by the grandchild's run-status
    sentinel so the audit tool can later attribute failures to specific
    pool instances / generations).
    """
    env = dict(parent_env)
    env["CARLA_POOL_MANAGED"] = "1"
    env["CARLA_POOL_HOST"] = str(lease.host)
    env["CARLA_POOL_WORLD_PORT"] = str(lease.world_port)
    env["CARLA_POOL_TM_PORT"] = str(lease.tm_port)
    env["CARLA_POOL_GENERATION"] = str(lease.generation)
    env["CARLA_POOL_INSTANCE_ID"] = str(lease.instance_id)
    env["CARLA_POOL_LEASE_ID"] = str(lease.lease_id)
    # Also set the standard CUDA_VISIBLE_DEVICES so planner weights
    # co-locate with the CARLA instance.
    env["CUDA_VISIBLE_DEVICES"] = str(lease.gpu_id)
    return env


# ----------------------------------------------------------------------
# Terminal-outcome classification
# ----------------------------------------------------------------------


def classify_lease_release_status(
    *,
    job_status: str,
) -> str:
    """Map a scheduler ``JobStatus`` to a ``CarlaPool.release_lease`` status.

    * SUCCESS / RETRY_QUEUED / RETRY_EXHAUSTED / SOFT_FAILURE -> "clean"
      (the grandchild exited normally; CARLA process is presumed healthy)
    * CARLA_CRASH / OOM                                       -> "crashed"
      (CARLA process or GPU state is compromised; force recycle)
    """
    if job_status in ("carla_crash", "oom"):
        return "crashed"
    return "clean"


# ----------------------------------------------------------------------
# Grandchild argv construction (Phase 3)
# ----------------------------------------------------------------------


# Options whose value must be stripped from the parent argv before we
# pass it along to a pool grandchild.  Replacements then re-inject the
# correct per-job values.
_POOL_VALUE_OPTIONS_TO_STRIP: tuple[str, ...] = (
    "--planner",
    "--gpu",
    "--gpus",
    "--port",
    "--traffic-manager-port",
    "--routes-dir",
    "--routes-leaf",
    "--scenario-name",
    "--results-tag",
    "--results-subdir",
    "--scenario-shard",
    "--scenario-retry-limit",
    "--zip",
    "--output-root",
    "--max-multiprocess",
    "--colmdriver-vllm-env",
    "--colmdriver-vllm-startup-timeout",
    "--carla-forensics-log-file",
    "--carla-forensics-server-log-file",
    "--planner-vram-estimate-mib",
    "--planner-vram-estimate-cache",
    # rootcause options: grandchild uses --no-start-carla so cannot use rootcause mode
    "--carla-rootcause-script",
    "--carla-rootcause-mode",
    "--carla-rootcause-diag-interval",
    "--carla-rootcause-startup-timeout",
    "--carla-rootcause-core-wait-seconds",
    "--carla-rootcause-log-root",
)


# Flag-only options that must be stripped (no value to pair with).
_POOL_FLAG_OPTIONS_TO_STRIP: tuple[str, ...] = (
    "--start-carla",
    "--no-start-carla",
    "--scenario-pool",
    "--carla-rootcause",
    "--no-scenario-pool",
    "--scenario-pool-no-reuse",
    "--overwrite",
    "--start-colmdriver-vllm",
    "--planner-dashboard",
    "--vram-aware-scheduler",
    "--no-vram-aware-scheduler",
)


def build_pool_grandchild_argv(
    *,
    base_argv: Sequence[str],
    job,                    # ScenarioJob
    lease,                  # CarlaLease
    results_subdir: str | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Construct argv for a pool grandchild.

    The grandchild is a ``run_custom_eval.py`` subprocess running in
    single-scenario mode against the pool-managed CARLA at
    ``lease.host:lease.world_port``.  It must NOT start its own CARLA,
    so ``--no-start-carla`` is injected and ``--start-carla`` stripped.
    """
    from tools.run_custom_eval import build_child_argv

    replacements: list[str] = [
        "--planner",
        job.planner_name,
        "--routes-dir",
        str(job.scenario_dir),
        "--port",
        str(lease.world_port),
        "--traffic-manager-port",
        str(lease.tm_port),
        "--results-tag",
        str(job.results_tag),
        "--no-start-carla",
        "--scenario-retry-limit",
        "1",   # retries are handled by the scheduler, not the grandchild
    ]
    if results_subdir:
        replacements.extend(
            [
                "--results-subdir",
                str(results_subdir),
            ]
        )
    if overwrite:
        replacements.append("--overwrite")

    return build_child_argv(
        base_argv,
        replacements=replacements,
        value_options_to_remove=_POOL_VALUE_OPTIONS_TO_STRIP,
        flag_options_to_remove=_POOL_FLAG_OPTIONS_TO_STRIP,
    )


# ----------------------------------------------------------------------
# Outcome classification (Phase 3)
# ----------------------------------------------------------------------


_TRANSIENT_FAILURE_KINDS: tuple[str, ...] = (
    # Grandchild's own run_status.json failure_kind values that are worth
    # retrying at the scheduler level.  We deliberately skip terminal markers
    # like ``no_route_execution`` / ``no_results_written`` / ``planner_env_error``
    # since those indicate the planner never made meaningful progress and
    # retrying yields the same outcome.
    "sensor_failure",
    "stalled",
    "carla_crash",
    "carla_unreachable",
    "carla_process_down",
    "rpc_timeout",
    "generic",  # SIGKILL-with-no-specific-marker often recovers on retry
)


def _tail_has_planner_route_command_none(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if "[tcp_route_diag]" in lowered and "next_cmd_is_none" in lowered:
        return True
    return (
        "'nonetype' object has no attribute 'value'" in lowered
        and ("tcp_agent.py" in lowered or "next_command" in lowered)
    )


# Deterministic failure signatures.  When one of these appears in the
# grandchild output (or in the run-status sentinel) the outcome is a
# scenario-data or code-path bug that retrying cannot fix -- re-enqueuing
# just wastes a CARLA slot for another 1-10 minutes.  Matching any of
# these forces ``soft_failure`` regardless of retry budget, and the
# reason is attached to the decision log so operators know *why* it was
# not retried.  The matching is case-insensitive substring.
_DETERMINISTIC_FAILURE_MARKERS: tuple[tuple[str, str], ...] = (
    # Scenario-data: the route XML could not be densified
    # NOTE: "ego route alignment failed" disabled — this fires on transient
    # CARLA RPC timeouts during the pre-scenario alignment step. The basic
    # perception_swap pool ran the same scenarioset successfully when these
    # warnings were present. Treating it as deterministic locks out cells
    # that would succeed on a fresh CARLA. Re-enable only if you see
    # genuinely-broken route XMLs that fail every retry regardless of CARLA.
    # ("ego route alignment failed", "scenario_data:alignment_failed"),
    ("dense trace too short", "scenario_data:dense_trace_too_short"),
    ("refusing to run the scenario on unaligned routes", "scenario_data:alignment_failed"),
    # Scenario-data: ego XMLs don't cover the requested ego count
    ("missing ego route xml for ego_id", "scenario_data:missing_ego_xml"),
    # Scenario-data: two egos in the same scenario request the same /
    # near-identical spawn coordinates (often an artifact of the dense
    # aligner snapping nearby ego start waypoints to the same lane
    # point).  The leaderboard accepts the partial spawn and runs to a
    # crash; retrying produces the same spawn collision.  With the
    # pre-scenario load_world sweep in place, lingering ghost actors
    # from a previous tenant should no longer be the source of partial
    # spawns, so this marker is safe to treat as terminal.
    ("accepting partial ego spawn", "scenario_data:partial_ego_spawn"),
    # Code-path: PDM metrics tripping on a nested-trajectory value
    ("typeerror: float() argument must be a string or a number, not 'location'",
     "planner_code:pdm_location_typeerror"),
    # Planner environment: the planner's Python import graph is broken
    # (missing package, wrong conda env).  Retrying can't fix it.
    # NOTE: "modulenotfounderror:" disabled — also fires when codriving's
    # subprocess hits a CUDA-init transient that surfaces the symptom
    # later in the import chain. Locking out blocks recoverable runs
    # (e.g. 2023-04-04-14-33-53_50_0/codriving). Other 3 markers below
    # remain active because they're more specific.
    # ("modulenotfounderror:", "planner_env:module_not_found"),
    # ImportError must be more specific to avoid matching warning-level log noise
    ("importerror: cannot import name", "planner_env:import_error"),
    ("no module named ", "planner_env:module_not_found"),
    # Scenario-data: route file references XML paths that do not exist
    ("no route xml files found under", "scenario_data:no_route_xml"),
)


def _find_deterministic_failure(tail_text: str) -> tuple[str, str] | None:
    """Return (marker, classification_tag) if ``tail_text`` matches a
    known non-retryable signature, else None."""
    lower = str(tail_text or "").lower()
    if not lower:
        return None
    for marker, tag in _DETERMINISTIC_FAILURE_MARKERS:
        if marker in lower:
            return marker, tag
    return None


def _append_pool_decision_log(log_path: Path | None, payload: dict[str, Any]) -> None:
    rendered = "[POOL DECISION] " + json.dumps(payload, sort_keys=True)
    log.info("scenario_pool: %s", rendered)
    if log_path is None:
        return
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    except Exception:
        log.exception("scenario_pool: failed to append decision log path=%s", log_path)


def _is_transient_sentinel_kind(kind: str) -> bool:
    k = str(kind or "").strip().lower()
    if not k:
        return False
    return k in _TRANSIENT_FAILURE_KINDS


def _grandchild_was_killed_by_signal(returncode: int) -> bool:
    """Heuristic: a ``Popen.wait()`` returncode < 0 means the child died to
    a signal (``-SIGNUM``).  That in turn means its ``atexit``/finally
    cleanup handlers did NOT run, so the warm CARLA server it was using
    now has orphan actors (ego + sensors + NPCs) in it.  We must recycle
    that instance before handing it to the next scenario, otherwise the
    next scenario's ``try_spawn_actor`` call will collide with stale
    actors and hard-fail with "Unable to spawn vehicle ... at ...".
    """
    try:
        return int(returncode) < 0
    except Exception:
        return False


def _scan_log_file_for_deterministic_marker(
    log_path: Path | None,
    *,
    max_bytes: int = 16 * 1024 * 1024,
) -> tuple[str, str] | None:
    """Scan the full per-scenario log file for a known-deterministic
    marker.  Returns ``(marker, tag)`` on match, else ``None``.

    Rationale: the in-memory ``output_tail`` passed into
    :func:`classify_grandchild_outcome` is capped at the last ~20 lines
    -- enough for most signatures but not for the "Accepting partial
    ego spawn" case, where the marker is printed once near the top of
    the run (~line 120) and then the scenario churns through 1000+
    destroy-actor log lines and sensor output before exiting.  By that
    point the marker has scrolled out of the tail buffer and the outer
    classifier falls through to the catchall-retry path, wasting
    ~500s per bad scenario.

    We bound the scan at 16 MiB so a pathologically verbose run can't
    block the scheduler thread.  Log files over that cap are unusual
    (the biggest one in the current baseline is ~6 MiB).
    """
    if log_path is None:
        return None
    try:
        log_path = Path(log_path)
        if not log_path.is_file():
            return None
        size = log_path.stat().st_size
        # If the file is smaller than max_bytes, read the whole thing;
        # otherwise read the HEAD (markers fire early, not late).
        with log_path.open("rb") as fh:
            if size <= max_bytes:
                data = fh.read()
            else:
                data = fh.read(max_bytes)
    except Exception:
        return None
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return None
    return _find_deterministic_failure(text)


def classify_grandchild_outcome(
    *,
    returncode: int,
    output_tail: Sequence[str],
    result_root: Path,
    retry_attempt: int = 0,
    retry_budget: int = 0,
    log_path: Path | None = None,
) -> tuple[str, str, str, str]:
    """Classify a grandchild outcome.

    Returns ``(job_status, carla_release_status)`` where:
      * ``job_status`` is one of ``success | retry_queued | soft_failure |
        carla_crash | oom`` (the string value of
        :class:`tools._scheduler.JobStatus`).
      * ``carla_release_status`` is ``clean | dirty | crashed``, passed
        straight to :meth:`CarlaPool.release_lease`.
      * ``decision_reason`` / ``decision_detail`` capture why the classifier
        chose that outcome so retry behavior is auditable in the pool log.

    When ``retry_attempt < retry_budget`` and the failure signature looks
    transient (sensor timeout, stall, carla crash/disconnect, SIGKILL
    without specific markers), the outcome is ``retry_queued`` so the
    scheduler re-enqueues the job on a fresh CARLA/GPU instead of burning
    the attempt as a partial.  Without this path the grandchild's own
    ``--scenario-retry-limit 1`` would turn every flaky failure into a
    permanent partial result.

    Preference order:
      1. Return code 0 -> success + clean.
      2. Evidence of OOM in output tail -> oom + crashed (no retry; the
         scheduler's OOM-multiplier knockdown is the correct response).
      3. Evidence of CARLA connectivity loss / crash -> carla_crash +
         crashed, retried if budget remains.
      4. Transient sentinel-recorded kind -> retry if budget remains, else
         soft_failure.
      5. Otherwise -> soft_failure + clean (CARLA itself looks healthy).
    """
    if int(returncode) == 0:
        return ("success", "clean", "returncode_zero", "grandchild exited cleanly")

    can_retry = int(retry_attempt) < int(retry_budget)
    # If the grandchild exited via signal, its cleanup handlers never ran,
    # so orphan CARLA actors from its scenario are still in the warm
    # world.  Every non-crashed release path below must be upgraded to
    # crashed in this case; we apply that override at the single exit
    # point below.
    force_recycle_from_signal = _grandchild_was_killed_by_signal(returncode)

    def _finish(
        status: str,
        release: str,
        reason: str,
        detail: str,
    ) -> tuple[str, str, str, str]:
        if force_recycle_from_signal and release != "crashed":
            # Primary root-cause fix for the ego-spawn collision cascade
            # observed in pool mode: a SIGKILL'd grandchild leaves actors
            # on the warm CARLA world, so the next scenario's ego spawn
            # lands on top of a stale ego/sensor and fails with "Unable
            # to spawn vehicle ... at ...".  Recycle the instance.
            return (status, "crashed", reason, detail)
        return (status, release, reason, detail)

    tail_joined = "\n".join(str(line) for line in (output_tail or []))
    tail_lower = tail_joined.lower()

    if _tail_has_planner_route_command_none(tail_lower):
        return _finish(
            "soft_failure",
            "clean",
            "planner_route_command_none",
            "deterministic TCP next_cmd=None signature in grandchild output",
        )

    # Deterministic scenario-data / planner-code failures.  Retrying these
    # wastes a CARLA slot for the same result.  Short-circuit to
    # soft_failure + clean.  ``clean`` (not ``crashed``) because nothing in
    # CARLA itself went wrong -- the grandchild's exit was caused by bad
    # input data or a planner bug, so the warm instance is safe to reuse.
    #
    # Two-pass search:
    #   1. Fast path: check the in-memory tail (captures most cases,
    #      especially planner/code crashes that fire shortly before exit).
    #   2. Slow path: if the tail didn't match but the caller provided a
    #      log_path, scan the whole file.  Catches "Accepting partial ego
    #      spawn" which fires near the top of the log and then scrolls
    #      out of the tail during a long run.
    det_hit = _find_deterministic_failure(tail_joined)
    if det_hit is None and log_path is not None:
        det_hit = _scan_log_file_for_deterministic_marker(log_path)
    if det_hit is not None:
        marker, tag = det_hit
        return _finish(
            "soft_failure",
            "clean",
            f"deterministic:{tag}",
            f"matched non-retryable marker={marker!r}",
        )

    # OOM heuristics: CUDA out-of-memory, nvidia-smi saturation, killed by
    # OS with signal=9 after large-allocation logs.  OOMs are *not* retried
    # at this layer -- the scheduler's OOM multiplier already throttles the
    # GPU, and a bare re-enqueue would just repeat the failure.
    oom_markers = (
        "CUDA out of memory",
        "out of memory",
        "OOM",
        "torch.cuda.OutOfMemoryError",
        "RuntimeError: CUDA",
    )
    if any(m in tail_joined for m in oom_markers):
        return _finish("oom", "crashed", "oom_marker", "OOM markers found in output tail")

    # CARLA crash / disconnect heuristics.
    crash_markers = (
        "RPCError",
        "rpc::timeout",
        "rpc::rpc_error",
        "connection refused",
        "ConnectionResetError",
        "BrokenPipeError",
        "CarlaUE4 crashed",
        "carla_world.tick",
        "carla_unreachable",
        "managed carla became unavailable",
    )
    if any(m in tail_joined for m in crash_markers):
        if can_retry:
            return _finish(
                "retry_queued",
                "crashed",
                "crash_marker_retry",
                "CARLA/RPC crash marker found and retry budget remains",
            )
        return _finish(
            "carla_crash",
            "crashed",
            "crash_marker_terminal",
            "CARLA/RPC crash marker found and retry budget exhausted",
        )

    sentinel_kind = ""
    try:
        sentinel = Path(result_root) / "_run_status.json"
        if sentinel.is_file():
            data = json.loads(sentinel.read_text())
            sentinel_kind = str(data.get("failure_kind", "")).lower()
    except Exception:
        sentinel_kind = ""

    if sentinel_kind:
        if sentinel_kind == "planner_route_command_none":
            return _finish(
                "soft_failure",
                "clean",
                "sentinel_planner_route_command_none",
                "grandchild sentinel recorded deterministic next_cmd=None failure",
            )
        if "oom" in sentinel_kind or "cuda" in sentinel_kind:
            return _finish("oom", "crashed", "sentinel_oom", f"sentinel failure_kind={sentinel_kind}")
        if (
            "carla" in sentinel_kind
            or "rpc" in sentinel_kind
            or "startup" in sentinel_kind
        ):
            if can_retry:
                return _finish(
                    "retry_queued",
                    "crashed",
                    "sentinel_carla_retry",
                    f"sentinel failure_kind={sentinel_kind} and retry budget remains",
                )
            return _finish(
                "carla_crash",
                "crashed",
                "sentinel_carla_terminal",
                f"sentinel failure_kind={sentinel_kind} and retry budget exhausted",
            )
        if _is_transient_sentinel_kind(sentinel_kind) and can_retry:
            # Sensor-failure sentinels usually mean the planner-side exit
            # was clean and CARLA is fine (scenario itself flaked); for
            # other transient kinds err toward crashed.  ``_finish`` will
            # still upgrade to crashed if the grandchild was signal-killed.
            release = "clean" if sentinel_kind == "sensor_failure" else "crashed"
            return _finish(
                "retry_queued",
                release,
                "sentinel_transient_retry",
                f"sentinel failure_kind={sentinel_kind} is retry-eligible",
            )

    # An unclassified non-zero exit is usually a SIGKILL via the parent
    # watchdog after CARLA stopped ticking.  Retry once if budget remains.
    if can_retry:
        return _finish(
            "retry_queued",
            "crashed",
            "catchall_retry",
            "non-zero exit with no non-retryable marker; using catch-all retry path",
        )
    return _finish(
        "soft_failure",
        "clean",
        "catchall_terminal",
        "non-zero exit with no retry budget remaining and no stronger classifier match",
    )


# ----------------------------------------------------------------------
# Driver: entry point invoked from run_custom_eval.main() when
# --scenario-pool is enabled.
# ----------------------------------------------------------------------


def run_scenario_pool(
    *,
    args,                                  # parsed argparse.Namespace from run_custom_eval
    planner_requests: list,                # list[PlannerRequest]
    vram_estimator,                        # PlannerVramEstimator
    gpu_monitor,                           # GPUMemoryMonitor
    planner_gpu_requests: list[int],
    planner_base_results_tag: str,
    planner_results_root: Path,
    repo_root: Path,
    carla_root: Path,
    colmdriver_vllm_manager=None,          # ColmDriverVLLMManager | None
    deferred_colmdriver_planner_names: Optional[set[str]] = None,
) -> int:
    """Build and drive the scenario-pool scheduler end-to-end.

    Returns a process exit code (0 on success, non-zero on hard failure).

    As of Phase 3, this function drives a working end-to-end dispatch:
    the scheduler acquires pool leases and spawns a ``run_custom_eval.py``
    grandchild per (planner, scenario) against each leased CARLA.  The
    grandchild is invoked with ``--no-start-carla`` plus the pool's
    port/TM-port, so it reuses the pool-managed CARLA instead of booting
    a fresh one.
    """
    from tools import _carla_pool as cp
    from tools import _scheduler as sch
    from tools.run_custom_eval import (
        _build_conda_launch_env,
        combine_results_subdir,
        detect_scenario_subfolders,
        MonitoredSubprocessRunner,
        pool_pre_scenario_world_sweep,
    )

    routes_dir = getattr(args, "routes_dir", None)
    if routes_dir is None:
        log.error(
            "--scenario-pool currently requires --routes-dir; "
            "scenario-zip mode is not yet supported in the pool path."
        )
        return 2
    routes_dir = Path(routes_dir).expanduser().resolve()
    if not routes_dir.exists():
        log.error("--scenario-pool: routes_dir does not exist: %s", routes_dir)
        return 2

    scenario_entries = detect_scenario_subfolders(
        routes_dir, getattr(args, "routes_leaf", None)
    )
    if not scenario_entries:
        log.error(
            "--scenario-pool: no scenario subfolders detected under %s",
            routes_dir,
        )
        return 2

    # Build the job list (planner x scenario) up front.  The pool path
    # owns retries at the scheduler level: the grandchild is always run
    # with ``--scenario-retry-limit 1`` (one attempt, no inner retries) so
    # we can re-enqueue it on a different CARLA on failure.  ``retry_budget``
    # here is the *number of retries after the first attempt*: 1 means one
    # retry (two attempts total).  Default to 1 in pool mode because the
    # previous single-attempt-only policy was turning transient ego-spawn
    # races / sensor timeouts / CARLA heartbeat freezes into permanent
    # partial results.
    raw_retry_budget = getattr(args, "scenario_retry_limit", None)
    if raw_retry_budget in (None, 0):
        retry_budget = 1
    else:
        retry_budget = max(0, int(raw_retry_budget) - 1)
    (
        pending_entries_by_planner,
        candidate_counts_by_planner,
        skipped_counts_by_planner,
    ) = filter_pending_pool_scenario_entries(
        planner_requests=planner_requests,
        scenario_entries=scenario_entries,
        repo_root=repo_root,
        results_tag_base=planner_base_results_tag,
        args=args,
    )
    total_candidate_jobs = sum(candidate_counts_by_planner.values())
    total_skipped_jobs = sum(skipped_counts_by_planner.values())
    jobs = build_scenario_job_list(
        planner_requests=planner_requests,
        scenario_entries=scenario_entries,
        scenario_entries_by_planner=pending_entries_by_planner,
        results_root=planner_results_root,
        results_tag_base=planner_base_results_tag,
        retry_budget=retry_budget,
        vram_estimator=vram_estimator,
    )
    if not jobs:
        log.info(
            "scenario_pool: all %d planner/scenario jobs already have completed results; nothing to queue",
            total_candidate_jobs,
        )
        print(
            "[INFO] Scenario pool: all planner/scenario jobs already have "
            "completed results; nothing to run.",
            flush=True,
        )
        return 0
    log.info(
        "scenario_pool: built %d pending jobs (skipped %d completed) across %d planners x %d discovered scenarios",
        len(jobs),
        total_skipped_jobs,
        len(planner_requests),
        len(scenario_entries),
    )
    for planner in planner_requests:
        planner_run_id = str(planner.run_id or planner.name)
        log.info(
            "scenario_pool: planner=%s pending=%d skipped_completed=%d discovered=%d",
            planner_run_id,
            len(pending_entries_by_planner.get(planner_run_id, [])),
            skipped_counts_by_planner.get(planner_run_id, 0),
            candidate_counts_by_planner.get(planner_run_id, 0),
        )
    reporter = _PoolStatusReporter(total_jobs=len(jobs))
    # Start the periodic dashboard.  Default 60s interval, configurable via
    # CARLA_POOL_DASHBOARD_INTERVAL_S.  Set to 0 to disable.
    try:
        _dash_interval = float(os.environ.get("CARLA_POOL_DASHBOARD_INTERVAL_S", "60"))
    except (TypeError, ValueError):
        _dash_interval = 60.0
    if _dash_interval > 0:
        reporter.start_dashboard(interval_s=_dash_interval)
    # Stuck killer runs on its own daemon (not coupled to dashboard) so a
    # slow dashboard render can't delay kills.  Uses CARLA_POOL_STUCK_KILL_S
    # threshold (default 600s = 10 min).  Disable via CARLA_POOL_STUCK_KILL_DISABLE=1.
    try:
        _killer_interval = float(os.environ.get("CARLA_POOL_STUCK_KILLER_INTERVAL_S", "60"))
    except (TypeError, ValueError):
        _killer_interval = 60.0
    if _killer_interval > 0:
        reporter.start_stuck_killer(interval_s=_killer_interval)
    reporter.emit_snapshot(
        "pool_initialized",
        detail=(
            f"planners={len(planner_requests)} scenarios={len(scenario_entries)} "
            f"jobs={len(jobs)} skipped_completed={total_skipped_jobs}"
        ),
    )

    # Construct the CARLA pool.  Reuse is honored per --scenario-pool-no-reuse;
    # the grandchild's own agent-setup flow loads the target town and spawns
    # fresh actors, so the pool does not need a defensive world reset between
    # leases.  A pure-connectivity sanity check is done in
    # ``CarlaPool.acquire_lease`` before handing the instance over.
    no_reuse = bool(getattr(args, "scenario_pool_no_reuse", False))
    idle_recycle_seconds = float(
        getattr(args, "pool_idle_recycle_seconds", 180.0) or 0.0
    )
    pool = cp.CarlaPool(
        carla_root=Path(carla_root),
        no_reuse=no_reuse,
        quiet=True,
        status_callback=reporter.on_carla_event,
        idle_recycle_seconds=idle_recycle_seconds,
    )
    log.info(
        "scenario_pool: pool configured no_reuse=%s (each release %s)",
        no_reuse,
        "recycles" if no_reuse else "returns to CLEAN for reuse",
    )
    reporter.emit_snapshot(
        "pool_configured",
        detail=f"no_reuse={no_reuse}",
    )

    base_argv = list(getattr(args, "_normalized_argv", sys.argv[1:]))
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_custom_eval.py"
    if not script_path.is_file():
        # Fallback: rely on sys.modules['tools.run_custom_eval'] file path.
        import tools.run_custom_eval as _rce_mod
        script_path = Path(_rce_mod.__file__).resolve()

    scenario_stall_output_timeout = float(
        getattr(args, "scenario_stall_output_timeout", 600.0) or 600.0
    )
    overwrite = bool(getattr(args, "overwrite", False))

    # Partition jobs: colmdriver/colmdriver_rulebase need vLLM running before
    # they can dispatch.  When --start-colmdriver-vllm + --defer-colmdriver-vllm-start
    # is in effect, run_custom_eval.main() created a ColmDriverVLLMManager but
    # did NOT start it (deferred until non-colmdriver planners drain).  Without
    # this gating, the pool scheduler would happily dispatch colmdriver scenarios
    # via WFQ, the agent would fail to connect to ports 1111/8888, and every such
    # job would burn ~20s + a partial result dir before being marked failed.
    # Defined here (before scheduler/closure setup) so _phase_state and
    # _on_result can reference these names without UnboundLocalError.
    _deferred_names = set(deferred_colmdriver_planner_names or set())
    _vllm_phase_jobs = [
        j for j in jobs if j.planner_name in _deferred_names
    ]
    _initial_jobs = [
        j for j in jobs if j.planner_name not in _deferred_names
    ]
    _vllm_phase_pending = bool(_vllm_phase_jobs) and colmdriver_vllm_manager is not None

    def _spawn_grandchild(*, job, lease):
        """Real Phase-3 grandchild spawn.

        Launches a ``run_custom_eval.py`` subprocess in single-scenario
        mode against the pool-provided CARLA.  Returns a
        :class:`ScenarioJobResult`.
        """
        scenario_results_subdir = combine_results_subdir(
            getattr(args, "results_subdir", None),
            job.scenario_name,
        )
        argv = build_pool_grandchild_argv(
            base_argv=base_argv,
            job=job,
            lease=lease,
            results_subdir=scenario_results_subdir,
            overwrite=overwrite,
        )
        conda_env = str(job.spawn_kwargs.get("conda_env") or "").strip()
        if conda_env:
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                conda_env,
                "python",
                str(script_path),
                *argv,
            ]
        else:
            cmd = [sys.executable, str(script_path), *argv]

        base_child_env = os.environ.copy()
        if conda_env:
            base_child_env = _build_conda_launch_env(base_child_env)
        child_env = build_grandchild_pool_env(parent_env=base_child_env, lease=lease)

        result_root = planner_results_root / job.planner_run_id / scenario_results_subdir
        log_slug = str(job.scenario_name).replace("/", "__").replace(os.sep, "__").replace(" ", "_")
        pool_log_path = (
            planner_results_root
            / "_planner_logs"
            / f"pool_{job.planner_run_id}__{log_slug}.log"
        )

        start = time.monotonic()
        reporter.on_eval_start(job=job, lease=lease, log_path=pool_log_path)
        sweep_counts = pool_pre_scenario_world_sweep(
            lease.host,
            lease.world_port,
            timeout_s=5.0,
            quiet=True,
            target_town=str(getattr(job, "town", "") or "") or None,
        )
        if int(sweep_counts.get("initial", 0) or 0) > 0 or not sweep_counts.get("ok", False):
            log.info(
                "scenario_pool: pre-scenario sweep job=%s carla=%s host=%s:%s "
                "ok=%s destroyed=%s/%s remaining=%s failures=%s load_world=%s "
                "rpc_attempts=%s",
                job.job_id,
                lease.instance_id,
                lease.host,
                lease.world_port,
                sweep_counts.get("ok", False),
                sweep_counts.get("destroyed", 0),
                sweep_counts.get("initial", 0),
                sweep_counts.get("remaining", 0),
                sweep_counts.get("failures", 0),
                sweep_counts.get("load_world_used", 0),
                sweep_counts.get("rpc_attempts", 0),
            )

        # If the sweep / RPC probe failed, the CARLA instance is unhealthy.
        # Return a ``retry_queued`` result that does NOT consume a retry
        # attempt, and mark the release as ``crashed`` so the pool
        # recycles the instance instead of handing the wreckage to the
        # next scenario.  The scheduler's enqueue_front() puts the job
        # back on its planner sub-queue with ``retry_attempt`` unchanged
        # -- we achieve "free requeue" by stamping the returned result
        # with the *same* retry_attempt the job came in with, so the next
        # grandchild attempt starts with the full budget.
        if not sweep_counts.get("ok", False):
            log.warning(
                "scenario_pool: CARLA unhealthy before dispatch job=%s carla=%s "
                "host=%s:%s -- recycling and requeueing (no retry burned)",
                job.job_id,
                lease.instance_id,
                lease.host,
                lease.world_port,
            )
            return sch.ScenarioJobResult(
                job=job,
                status=sch.JobStatus.RETRY_QUEUED,
                result_root=result_root,
                duration_s=time.monotonic() - start,
                lease_gpu_id=lease.gpu_id,
                carla_instance_id=lease.instance_id,
                carla_generation=lease.generation,
                carla_release_status="crashed",
                stderr_tail=(
                    f"pre_scenario_sweep_failed "
                    f"rpc_attempts={sweep_counts.get('rpc_attempts', 0)} "
                    f"remaining_actors={sweep_counts.get('remaining', 0)} "
                    f"load_world_used={sweep_counts.get('load_world_used', 0)}"
                ),
                decision_reason="pre_scenario_sweep_failed",
            )
        try:
            run_result = MonitoredSubprocessRunner(
                cmd,
                env=child_env,
                cwd=repo_root,
                timeout_s=None,
                stall_output_timeout_s=scenario_stall_output_timeout,
                output_prefix=f"[pool:{job.planner_run_id}:{job.scenario_name}] ",
                show_carla_diag=False,
                emit_stdout=False,
                log_path=pool_log_path,
            ).run()
        except Exception as exc:
            log.exception(
                "scenario_pool: grandchild spawn raised job=%s", job.job_id
            )
            return sch.ScenarioJobResult(
                job=job,
                status=sch.JobStatus.SOFT_FAILURE,
                result_root=result_root,
                duration_s=time.monotonic() - start,
                lease_gpu_id=lease.gpu_id,
                carla_instance_id=lease.instance_id,
                carla_generation=lease.generation,
                carla_release_status="crashed",
                stderr_tail=f"{type(exc).__name__}: {exc}",
                decision_reason="grandchild_spawn_exception",
            )

        duration = time.monotonic() - start
        output_tail = list(getattr(run_result, "output_tail", []) or [])
        returncode = int(getattr(run_result, "returncode", -1))

        (
            job_status_str,
            carla_release_status,
            decision_reason,
            decision_detail,
        ) = classify_grandchild_outcome(
            returncode=returncode,
            output_tail=output_tail,
            result_root=result_root,
            retry_attempt=int(getattr(job, "retry_attempt", 0) or 0),
            retry_budget=int(getattr(job, "retry_budget", 0) or 0),
            log_path=pool_log_path,
        )
        status = sch.JobStatus(job_status_str)
        _append_pool_decision_log(
            pool_log_path,
            {
                "planner": job.planner_name,
                "scenario": job.scenario_name,
                "job_id": job.job_id,
                "retry_attempt": int(getattr(job, "retry_attempt", 0) or 0),
                "retry_budget": int(getattr(job, "retry_budget", 0) or 0),
                "status": status.value,
                "carla_release_status": carla_release_status,
                "decision_reason": decision_reason,
                "decision_detail": decision_detail,
                "returncode": returncode,
                "terminated_reason": getattr(run_result, "terminated_reason", None),
                "child_pid": getattr(run_result, "pid", None),
                "signal_num": getattr(run_result, "signal_num", None),
                "signal_name": getattr(run_result, "signal_name", None),
                "elapsed_s": round(float(getattr(run_result, "elapsed_s", 0.0) or 0.0), 3),
                "last_output_age_s": round(
                    float(getattr(run_result, "last_output_age_s", 0.0) or 0.0),
                    3,
                ),
                "last_any_output_age_s": round(
                    float(getattr(run_result, "last_any_output_age_s", 0.0) or 0.0),
                    3,
                ),
            },
        )

        # When the scheduler will re-enqueue this job, wipe the partial
        # result tree written by the grandchild so the retry starts from a
        # clean slate and the finalize_failed_result_roots rename
        # ("_partial_<ts>") isn't applied to what is really an in-flight
        # attempt.  Leaving the half-written directory in place makes the
        # retry think the scenario already ran.
        if status == sch.JobStatus.RETRY_QUEUED:
            try:
                import shutil
                if result_root.exists():
                    shutil.rmtree(result_root, ignore_errors=True)
            except Exception:
                log.exception(
                    "scenario_pool: could not clean result_root for retry job=%s",
                    job.job_id,
                )

        return sch.ScenarioJobResult(
            job=job,
            status=status,
            result_root=result_root,
            duration_s=duration,
            lease_gpu_id=lease.gpu_id,
            carla_instance_id=lease.instance_id,
            carla_generation=lease.generation,
            carla_release_status=carla_release_status,
            stderr_tail="\n".join(output_tail[-20:]),
            decision_reason=decision_reason,
        )

    # Phase-transition state for deferred CoLMDriver vLLM startup.
    _phase_lock = threading.Lock()
    _phase_state = {
        "initial_remaining": len(_initial_jobs),
        "vllm_phase_started": False,
    }

    def _maybe_start_vllm_phase(scheduler_ref):
        """Background-thread entrypoint that boots vLLM and enqueues the
        deferred CoLMDriver jobs.  Called exactly once, after the last
        non-colmdriver job finishes.  Boots can take 60-180s, which is
        why this runs off the worker thread.  The scheduler is held open
        by an external hold acquired at startup; we always release it in
        the finally clause so a vLLM startup failure doesn't deadlock the
        run loop."""
        try:
            log.info(
                "colmdriver phase: non-colmdriver drain complete, starting "
                "vLLM services (this may take 60-180s)..."
            )
            print(
                "[INFO] CoLMDriver phase: starting vLLM services...",
                flush=True,
            )
            colmdriver_vllm_manager.start()
            log.info(
                "colmdriver phase: vLLM ready, enqueueing %d deferred jobs",
                len(_vllm_phase_jobs),
            )
            print(
                f"[INFO] CoLMDriver phase: vLLM up; releasing "
                f"{len(_vllm_phase_jobs)} deferred jobs.",
                flush=True,
            )
            for j in _vllm_phase_jobs:
                scheduler_ref.enqueue(j)
        except Exception:
            log.exception(
                "colmdriver phase: vLLM startup or enqueue failed; deferred "
                "jobs will not run this session"
            )
        finally:
            try:
                scheduler_ref.release_external_hold()
            except Exception:
                log.exception("colmdriver phase: failed to release external hold")

    def _on_result(result) -> None:
        log.info(
            "scenario_pool: job=%s status=%s duration=%.1fs gpu=%d",
            result.job.job_id,
            result.status.value,
            result.duration_s,
            result.lease_gpu_id,
        )
        reporter.on_eval_finish(result=result)
        # Emit a per-reason failure breakdown every 10 finishes so the
        # operator can see total failure attempts and what's dominating
        # without having to grep the log.
        try:
            total_finished = (
                getattr(reporter, "_evals_ok", 0)
                + getattr(reporter, "_evals_failed", 0)
                + getattr(reporter, "_evals_failed_attempts", 0)
            )
            if total_finished and total_finished % 10 == 0:
                reporter.emit_failure_breakdown()
        except Exception:
            pass

        # Phase-transition trigger: if this was a non-colmdriver job AND
        # only colmdriver jobs remain, kick off vLLM start + deferred enqueue.
        # Only fires for terminal statuses (not RETRY_QUEUED) to avoid
        # double-counting retry re-enqueues.
        if not _vllm_phase_pending:
            return
        try:
            from tools._scheduler import JobStatus as _JS
            terminal = result.status not in (_JS.RETRY_QUEUED,)
        except Exception:
            terminal = True
        if not terminal:
            return
        if result.job.planner_name in _deferred_names:
            return
        should_kick = False
        with _phase_lock:
            _phase_state["initial_remaining"] -= 1
            if (
                _phase_state["initial_remaining"] <= 0
                and not _phase_state["vllm_phase_started"]
            ):
                _phase_state["vllm_phase_started"] = True
                should_kick = True
        if should_kick:
            t = threading.Thread(
                target=_maybe_start_vllm_phase,
                args=(scheduler,),
                daemon=True,
                name="colmdriver-phase-start",
            )
            t.start()

    # Edge case: only colmdriver planners requested (no non-colmdriver jobs to
    # drain).  Kick the phase immediately so the deferred enqueue happens.
    if _vllm_phase_pending and not _initial_jobs:
        with _phase_lock:
            kick_now = not _phase_state["vllm_phase_started"]
            if kick_now:
                _phase_state["vllm_phase_started"] = True
        if kick_now:
            threading.Thread(
                target=_maybe_start_vllm_phase,
                args=(scheduler,),
                daemon=True,
                name="colmdriver-phase-start",
            ).start()

    # Wire the scheduler.  New knobs (all optional, safe defaults below) are
    # exposed to the CLI via --pool-max-leases-per-gpu /
    # --pool-oom-recovery-idle-s so we can tune deadlock-recovery behavior
    # without recompiling.
    # Reuse the operator-facing --scheduler-tick-log flag for pool mode
    # too.  When set, the scheduler writes one JSONL line per dispatch
    # decision / completion with GPU budgets, active leases, and the
    # chosen placement, making it trivial to see whether we're leaving
    # capacity on the table in steady state.
    raw_tick_log = getattr(args, "scheduler_tick_log", None)
    decision_log_path: Path | None = None
    if raw_tick_log:
        decision_log_path = Path(str(raw_tick_log)).expanduser()
        if not decision_log_path.suffix:
            decision_log_path = decision_log_path / "scheduler_ticks.jsonl"
        try:
            decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            log.exception(
                "scenario_pool: could not create decision log dir %s",
                decision_log_path.parent,
            )
            decision_log_path = None

    scheduler = sch.ScenarioScheduler(
        carla_pool=pool,
        gpu_monitor=gpu_monitor,
        vram_estimator=vram_estimator,
        spawn_grandchild=_spawn_grandchild,
        on_result=_on_result,
        gpu_ids=list(planner_gpu_requests),
        carla_vram_estimate_mib=int(getattr(args, "carla_vram_estimate_mib", 6000)),
        vram_safety_buffer_mib=int(getattr(args, "vram_safety_buffer_mib", 1500)),
        max_leases_per_gpu=int(getattr(args, "pool_max_leases_per_gpu", 4) or 0) or None,
        oom_multiplier_recovery_idle_s=float(
            getattr(args, "pool_oom_recovery_idle_s", 180.0) or 180.0
        ),
        decision_log_path=decision_log_path,
    )
    for planner in planner_requests:
        scheduler.register_planner(planner.run_id or planner.name)

    # Enqueue only the initial (non-colmdriver) jobs upfront.  The
    # _vllm_phase_jobs are held back and enqueued by _maybe_start_vllm_phase
    # once the non-colmdriver queues drain.  Partition was computed near the
    # top of this function so closures (_phase_state, _on_result) could
    # reference it.
    for job in _initial_jobs:
        scheduler.enqueue(job)
    if _vllm_phase_pending:
        # Hold the scheduler open across the gap between "last non-colmdriver
        # job finishes" and "deferred colmdriver jobs are enqueued".  Without
        # this, run() would see an empty queue + zero in-flight and exit
        # before the phase-start thread had a chance to boot vLLM.
        scheduler.acquire_external_hold()
        log.info(
            "scenario_pool: deferring %d colmdriver job(s) until %d non-colmdriver "
            "job(s) drain; vLLM start will happen at phase transition.",
            len(_vllm_phase_jobs), len(_initial_jobs),
        )
        print(
            f"[INFO] CoLMDriver phase deferred: {len(_vllm_phase_jobs)} jobs queued "
            f"behind {len(_initial_jobs)} non-colmdriver jobs; vLLM will start "
            f"after non-colmdriver drain.",
            flush=True,
        )

    # Install a SIGINT / SIGTERM handler that initiates an *orderly*
    # shutdown instead of leaking CARLAs and leaderboard_evaluator
    # grandchildren into init.  The prior behaviour (no explicit handler
    # + bare raise on KeyboardInterrupt) is how the machine ended up
    # with 21-hour-old leaderboard evaluators after an earlier run was
    # Ctrl-C'd.  Previous handlers are chained-restored on exit so this
    # is safe to run under nested-signal setups (e.g. pytest).
    import signal
    import os as _os_signal
    _prev_sigint = None
    _prev_sigterm = None
    _interrupt_reason: dict[str, str] = {"value": ""}
    # Track signal arrival times for the double-tap escape: if Ctrl+C is
    # hit twice within ``DOUBLE_TAP_WINDOW_S``, hard-exit immediately
    # instead of waiting for graceful drain.
    _signal_state: dict[str, float] = {"last_ts": 0.0}
    DOUBLE_TAP_WINDOW_S = 5.0

    def _on_signal(signum, _frame):  # noqa: ANN001 -- OS signature
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        now = time.monotonic()
        # Double-tap escape: a second Ctrl+C within 5s skips graceful
        # drain entirely and hard-exits.  Avoids the "Ctrl+C never wants
        # to stop" frustration when in-flight grandchildren are
        # legitimately busy and would otherwise need to time out.
        if (
            signum == signal.SIGINT
            and _signal_state["last_ts"] > 0
            and (now - _signal_state["last_ts"]) <= DOUBLE_TAP_WINDOW_S
        ):
            print(
                f"\n[POOL] Second {sig_name} within {DOUBLE_TAP_WINDOW_S:.0f}s — "
                "hard-exiting. CARLAs/grandchildren will be cleaned up by "
                "the OS via the process group; orphans may survive briefly.",
                flush=True,
            )
            try:
                # Send SIGKILL to the whole process group so descendants die
                # quickly even if their Python finallys are misbehaving.
                _os_signal.killpg(_os_signal.getpgid(0), signal.SIGKILL)
            except Exception:
                pass
            _os_signal._exit(130)
        _signal_state["last_ts"] = now

        _interrupt_reason["value"] = sig_name
        log.warning(
            "scenario_pool: received %s -- initiating graceful shutdown "
            "(press Ctrl+C again within %.0fs to hard-exit)",
            sig_name,
            DOUBLE_TAP_WINDOW_S,
        )
        try:
            reporter.emit_snapshot("pool_signal", detail=f"signal={sig_name}")
        except Exception:
            pass
        try:
            scheduler.shutdown()
        except Exception:
            log.exception("scenario_pool: scheduler.shutdown() raised in signal handler")
        # Aggressively SIGTERM every active grandchild (leaderboard_evaluator
        # subprocesses) so the worker threads' subprocess.wait() returns
        # quickly. Without this, the scheduler's drain loop blocks for
        # dispatch_tick_s per worker thread waiting for subprocesses that
        # nobody told to exit -- which is the "Ctrl+C never stops" bug.
        try:
            from tools.run_custom_eval import (
                _active_monitored_runners,
                _active_monitored_runners_lock,
            )
            with _active_monitored_runners_lock:
                runners = list(_active_monitored_runners)
            terminated = 0
            for runner in runners:
                try:
                    runner.terminate(reason=f"pool {sig_name} graceful shutdown")
                    terminated += 1
                except Exception:
                    pass
            if terminated:
                print(
                    f"[POOL] Terminated {terminated} in-flight grandchild "
                    f"subprocess(es) on {sig_name}; drain should complete shortly.",
                    flush=True,
                )
        except Exception:
            log.exception("scenario_pool: failed to terminate active grandchildren")

    def _on_drain_signal(_signum, _frame):  # noqa: ANN001
        """Graceful drain: stop accepting new dispatches, let in-flight
        complete naturally, then exit.  Operator triggers via:
            kill -USR1 <parent_pid>
        """
        log.warning("scenario_pool: received SIGUSR1 -- entering DRAIN MODE")
        try:
            reporter.emit_snapshot("pool_drain_started",
                                   detail="no_new_dispatches_will_be_accepted")
        except Exception:
            pass
        try:
            scheduler.enter_drain_mode()
        except AttributeError:
            log.warning("scheduler.enter_drain_mode() not available; falling back to shutdown")
            scheduler.shutdown()
        except Exception:
            log.exception("scenario_pool: enter_drain_mode raised")

    try:
        _prev_sigint = signal.signal(signal.SIGINT, _on_signal)
    except (ValueError, OSError):
        _prev_sigint = None
    try:
        _prev_sigterm = signal.signal(signal.SIGTERM, _on_signal)
    except (ValueError, OSError):
        _prev_sigterm = None
    _prev_sigusr1 = None
    try:
        _prev_sigusr1 = signal.signal(signal.SIGUSR1, _on_drain_signal)
    except (ValueError, OSError, AttributeError):
        # SIGUSR1 not available on Windows; safe to skip.
        _prev_sigusr1 = None

    try:
        scheduler.run()
        if _interrupt_reason["value"]:
            reporter.emit_snapshot(
                "pool_interrupted",
                detail=f"signal={_interrupt_reason['value']}",
            )
            return 130
        reporter.emit_snapshot("pool_complete", detail="scheduler_run_finished")
        return 0
    except KeyboardInterrupt:
        scheduler.shutdown()
        reporter.emit_snapshot("pool_interrupted", detail="keyboard_interrupt")
        raise
    except Exception:
        log.exception("scenario_pool: scheduler.run() raised")
        reporter.emit_snapshot("pool_failed", detail="scheduler_run_raised")
        return 1
    finally:
        # Restore prior signal handlers so e.g. a pytest runner that
        # imported us doesn't wind up with our handlers installed.
        try:
            if _prev_sigint is not None:
                signal.signal(signal.SIGINT, _prev_sigint)
            if _prev_sigterm is not None:
                signal.signal(signal.SIGTERM, _prev_sigterm)
        except (ValueError, OSError):
            pass
        try:
            pool.shutdown()
            reporter.emit_snapshot("pool_shutdown", detail="all_carlas_stopped")
            # Final dashboard + failure breakdown so the operator sees the
            # full picture even on Ctrl-C / abnormal exit.
            reporter.emit_dashboard()
            reporter.emit_failure_breakdown()
            reporter.stop_dashboard()
        except Exception:
            log.exception("scenario_pool: pool.shutdown() raised")
