"""Global scenario scheduler for the scenario-pool architecture.

Owns the work queue, VRAM accounting, and parallel dispatch of grandchild
evaluator processes.  Replaces the per-planner serial for-loop that lives
inside the current ``_planner_runner_main``.

Key design points
-----------------
* VRAM is the only hard cap.  Capacity is emergent: ``effective_budget[gpu]``
  = ``(free_mib - safety_buffer) * oom_multiplier``; we dispatch iff
  ``required_mib <= effective_budget[gpu]``.
* "Required" = ``planner_vram_estimate`` + (``carla_vram_estimate`` iff we
  also need to start a new CARLA for this job -- if a warm instance of the
  right town already exists, only the planner cost counts).
* Weighted round-robin across planner sub-queues, weight proportional to
  ``1/mean_run_scenario_s`` per planner, so fast planners don't starve
  behind slow ones.
* All retry accounting and all terminal-outcome finalization
  (``write_run_status`` + ``finalize_failed_result_roots``) is preserved
  verbatim -- the scheduler is a relocation of today's logic, not a
  rewrite of validity guarantees.

This module is scheduler-only; process lifecycle for CARLA is in
:mod:`tools._carla_pool`.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import os as _os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional
try:
    from typing import Literal  # Python 3.8+
except ImportError:
    from typing_extensions import Literal  # Python 3.7 fallback (colmdrivermarco2)


log = logging.getLogger(__name__)


class JobStatus(str, enum.Enum):
    """Terminal status of a scenario job as observed by the scheduler."""

    SUCCESS = "success"
    RETRY_QUEUED = "retry_queued"
    RETRY_EXHAUSTED = "retry_exhausted"
    SOFT_FAILURE = "soft_failure"
    CARLA_CRASH = "carla_crash"
    OOM = "oom"


@dataclasses.dataclass
class ScenarioJob:
    """One scenario, one planner.  Unit of work for the scheduler."""

    job_id: str
    planner_run_id: str
    planner_name: str
    scenario_name: str
    scenario_dir: Path
    results_tag: str
    town: str
    vram_estimate_mib: int
    retry_budget: int
    retry_attempt: int = 0
    # Counts the number of "free requeues" (infra-failure re-enqueues that
    # did not burn a retry attempt).  Capped by
    # ScenarioScheduler._max_free_requeues_per_job so a permanently bad
    # scenario or CARLA cluster can't loop a single job forever.
    free_requeues: int = 0
    enqueued_at: float = dataclasses.field(default_factory=time.monotonic)
    # Opaque payload threaded through to the grandchild spawn step.
    spawn_kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ScenarioJobResult:
    """Outcome of one scenario run, as seen by the scheduler."""

    job: ScenarioJob
    status: JobStatus
    result_root: Path
    duration_s: float
    lease_gpu_id: int
    carla_instance_id: str
    carla_generation: int
    carla_release_status: Literal["clean", "dirty", "crashed"]
    stderr_tail: str = ""
    # Failure-classifier output (catchall_retry, crash_marker_terminal,
    # deterministic:scenario_data:partial_ego_spawn, etc.).  Empty when status
    # is success.  Surfaced so the status reporter can break down the total
    # failure-attempts counter by reason without re-parsing the log.
    decision_reason: str = ""
    # When True and status == RETRY_QUEUED, the scheduler re-enqueues the
    # job without incrementing retry_attempt.  Used for infrastructure
    # failures the job never actually observed (e.g. the pre-scenario
    # CARLA sweep / liveness probe declared the instance unhealthy before
    # the grandchild ever ran).  Capped internally so a permanently bad
    # scenario cannot loop forever on free requeues.
    free_requeue: bool = False


@dataclasses.dataclass
class _PlannerQueue:
    """Per-planner sub-queue with a scheduling weight."""

    planner_name: str
    weight: float            # higher = more frequent dispatch
    jobs: list[ScenarioJob] = dataclasses.field(default_factory=list)
    # Virtual-time counter for weighted fair queueing (WFQ).  Each time
    # we dispatch from this queue we advance vtime by 1/weight, so the
    # queue with the lowest vtime is always the most "deserving".
    vtime: float = 0.0


@dataclasses.dataclass
class _GpuReservationWindow:
    """Tracks launch reservations that are not yet visible in GPU telemetry."""

    reserved_mib: int = 0
    baseline_used_mib: int | None = None
    peak_used_mib: int | None = None


class ScenarioScheduler:
    """Global scheduler.

    The scheduler holds a per-planner queue map and dispatches grandchildren
    concurrently against a :class:`tools._carla_pool.CarlaPool`.  It does
    NOT know about the hierarchy below it: every grandchild spawn is done
    via ``spawn_grandchild(job, lease)`` injected by the parent, so the
    scheduler is testable in isolation.
    """

    def __init__(
        self,
        *,
        carla_pool,                                      # tools._carla_pool.CarlaPool
        gpu_monitor,                                     # GPUMemoryMonitor from run_custom_eval
        vram_estimator,                                  # PlannerVramEstimator from run_custom_eval
        spawn_grandchild: Callable[..., ScenarioJobResult],
        on_result: Callable[[ScenarioJobResult], None],
        gpu_ids: list[int],
        carla_vram_estimate_mib: int,
        vram_safety_buffer_mib: int,
        dispatch_tick_s: float = 0.5,
        fresh_snapshot_max_age_s: float = 5.0,
        oom_backoff_factor: float = 0.8,
        oom_backoff_recover_count: int = 5,
        oom_multiplier_floor: float = 0.3,
        oom_multiplier_recovery_idle_s: float = 180.0,
        max_leases_per_gpu: int | None = 4,
        max_free_requeues_per_job: int = 3,
        decision_log_path: Optional[Path] = None,
    ) -> None:
        self._pool = carla_pool
        self._gpu_monitor = gpu_monitor
        self._estimator = vram_estimator
        self._spawn_grandchild = spawn_grandchild
        self._on_result = on_result
        self._gpu_ids = [int(g) for g in gpu_ids]
        self._carla_vram_estimate_mib = int(carla_vram_estimate_mib)
        self._vram_safety_buffer_mib = int(vram_safety_buffer_mib)
        self._dispatch_tick_s = float(dispatch_tick_s)
        self._fresh_snapshot_max_age_s = float(fresh_snapshot_max_age_s)
        self._oom_backoff_factor = float(oom_backoff_factor)
        self._oom_backoff_recover_count = int(oom_backoff_recover_count)
        self._oom_multiplier_floor = float(max(0.05, oom_multiplier_floor))
        self._oom_multiplier_recovery_idle_s = float(max(30.0, oom_multiplier_recovery_idle_s))
        # None or <=0 disables the concurrency cap.
        self._max_leases_per_gpu = (
            None if max_leases_per_gpu is None or int(max_leases_per_gpu) <= 0
            else int(max_leases_per_gpu)
        )
        self._max_free_requeues_per_job = max(0, int(max_free_requeues_per_job))
        self._decision_log_path = Path(decision_log_path) if decision_log_path else None
        self._decision_log_lock = threading.Lock()
        if self._decision_log_path is not None:
            try:
                self._decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                log.exception(
                    "scheduler: could not create decision log parent dir %s",
                    self._decision_log_path,
                )

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._queues: dict[str, _PlannerQueue] = {}
        self._in_flight: dict[str, ScenarioJob] = {}  # job_id -> job
        self._worker_threads: list[threading.Thread] = []
        self._shutdown = False
        # Worker-thread reaper diagnostics. Populated by _reap_dead_workers
        # and surfaced via _emit_scheduler_health.
        self._reaper_total_reaped: int = 0
        self._reaper_call_count: int = 0
        # Periodic health emission: confirms the thread-leak fix is working.
        self._health_last_emit_at: float = 0.0
        self._health_emit_interval_s: float = 30.0

        # Per-GPU dynamic budget knockdown after OOM.
        self._gpu_oom_multiplier: dict[int, float] = {g: 1.0 for g in self._gpu_ids}
        self._gpu_successes_since_oom: dict[int, int] = {g: 0 for g in self._gpu_ids}
        # Monotonic timestamp of the most recent OOM we observed on each GPU.
        # Used by the deadlock-recovery path: if no jobs have been in flight on
        # this GPU AND no OOM has been observed for
        # ``oom_multiplier_recovery_idle_s`` seconds, we gradually recover the
        # multiplier so the system doesn't permanently wedge when all small
        # planners drain and only large ones remain.
        self._gpu_last_oom_monotonic: dict[int, float] = {g: 0.0 for g in self._gpu_ids}
        # Active leases per GPU (for the concurrency cap).  Incremented in
        # ``_reserve_gpu_vram`` (dispatch-time), decremented in
        # ``_release_gpu_vram`` (worker finalize).
        self._gpu_active_leases: dict[int, int] = {g: 0 for g in self._gpu_ids}

        # Estimated VRAM consumption currently in flight per GPU.  Used to
        # avoid double-dispatching into free-but-not-yet-consumed headroom.
        # This is a *launch* reservation, not a full-job reservation: once
        # nvidia-smi has observed the memory rise, the reservation should stop
        # suppressing capacity or we double-count long-running jobs.
        self._gpu_reserved_mib: dict[int, int] = {g: 0 for g in self._gpu_ids}
        # Drain mode: when set via enter_drain_mode(), stop dispatching new
        # jobs but let in-flight complete naturally.  Triggered by SIGUSR1
        # to the parent.  Different from _shutdown which kills everything.
        self._drain_mode: bool = False
        # External holds keep run() alive even when queues are empty and
        # nothing is in flight.  Used by callers that plan to enqueue more
        # jobs after some external event (e.g. CoLMDriver vLLM startup).
        self._external_holds: int = 0
        # Per-GPU dispatch cooldown: prevents the OOM race where two
        # planners dispatch to the same GPU within nvidia-smi's update
        # window (~2-5s) and collectively exceed available VRAM.  Default
        # 5s cooldown.  Set CARLA_GPU_DISPATCH_COOLDOWN_S=0 to disable.
        self._gpu_last_dispatch_at: dict[int, float] = {g: 0.0 for g in self._gpu_ids}
        try:
            import os as _os
            self._gpu_dispatch_cooldown_s: float = max(
                0.0, float(_os.environ.get("CARLA_GPU_DISPATCH_COOLDOWN_S", "5"))
            )
        except (TypeError, ValueError):
            self._gpu_dispatch_cooldown_s = 5.0
        self._gpu_reservation_windows: dict[int, _GpuReservationWindow] = {
            g: _GpuReservationWindow() for g in self._gpu_ids
        }

        # Per-GPU CARLA-start failure tracking.  When a GPU accumulates N
        # consecutive start failures, it is temporarily excluded from
        # new-CARLA placement decisions (Priority 3 in _pick_gpu_and_decide_carla).
        # Existing CARLA instances on a blacklisted GPU are NOT affected;
        # they still get new leases via Priority 1/2.  The blacklist has an
        # exponential backoff that resets fully on any successful start.
        self._gpu_consecutive_start_failures: dict[int, int] = {g: 0 for g in self._gpu_ids}
        self._gpu_start_blocked_until: dict[int, float] = {g: 0.0 for g in self._gpu_ids}
        # How many consecutive failures before the first backoff kicks in.
        self._gpu_start_failure_threshold: int = 3
        # Base backoff in seconds (doubles every 3 additional failures).
        self._gpu_start_base_backoff_s: float = 120.0
        # Maximum blacklist duration (1 hour; GPU might recover after a restart).
        self._gpu_start_max_backoff_s: float = 3600.0

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def register_planner(self, planner_name: str, *, weight: float = 1.0) -> None:
        """Create a per-planner sub-queue.  Must be called before enqueue."""
        with self._lock:
            self._queues.setdefault(
                planner_name,
                _PlannerQueue(planner_name=planner_name, weight=max(1e-6, float(weight))),
            )

    def enqueue(self, job: ScenarioJob) -> None:
        """Add a job to its planner's sub-queue."""
        with self._cond:
            q = self._queues.get(job.planner_name)
            if q is None:
                raise KeyError(f"planner not registered: {job.planner_name}")
            q.jobs.append(job)
            self._cond.notify_all()

    def enqueue_front(self, job: ScenarioJob) -> None:
        """Re-enqueue at head (used for retries)."""
        with self._cond:
            q = self._queues[job.planner_name]
            q.jobs.insert(0, job)
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _reap_dead_workers(self) -> None:
        """Join completed worker threads so the OS reclaims their task entries.

        Without this, every dispatched scenario leaves a zombie thread behind
        in /proc/<master_pid>/task. Over a long-running pool (thousands of
        scenarios) the master process accumulates tens of thousands of
        zombies and eventually hits the per-process thread limit, at which
        point ``threading.Thread().start()`` raises ``RuntimeError("can't
        start new thread")``. We observed a master at 15,568 zombie threads
        before its sibling orchestrator died from this.

        ``Thread.join(timeout=0.001)`` on a thread that has already returned
        from its target is a near-instantaneous reap (just clears the OS task
        entry); for live threads it returns without blocking. So this is
        cheap to call every dispatch tick.
        """
        with self._lock:
            threads_snapshot = list(self._worker_threads)
        # Build the list of dead threads OUTSIDE the lock so is_alive() polls
        # don't hold up dispatch decisions.
        reaped: list = []
        for t in threads_snapshot:
            if not t.is_alive():
                try:
                    t.join(timeout=0.001)
                except Exception:
                    pass
                reaped.append(t)
        if reaped:
            with self._lock:
                # Defensive: another path may have removed entries; only drop
                # the ones we actually reaped.
                self._worker_threads[:] = [
                    t for t in self._worker_threads if t not in reaped
                ]
                self._reaper_total_reaped += len(reaped)
                self._reaper_call_count += 1

    def _emit_scheduler_health(self) -> None:
        """Emit a periodic scheduler-health event to the tick log.

        Surfaces the metrics that confirm the thread-leak fix is working:
          * master_thread_count -- /proc/<pid>/task count. The leak symptom
            was this climbing past 15,000 before the master crashed.
            With the fix this should plateau under ~100.
          * worker_threads_tracked -- size of self._worker_threads (the list
            the reaper drains). Should hover near self._in_flight size.
          * probe_cache stats -- hit/miss/eviction counts. Hits should
            dominate (>= 95% of probe attempts after warmup).
          * reaper stats -- total reaped count + call count.

        Throttled to once every ~30s; cheap enough to call every dispatch
        tick if needed.
        """
        now = time.monotonic()
        if (now - self._health_last_emit_at) < self._health_emit_interval_s:
            return
        self._health_last_emit_at = now

        # Master /proc thread count -- the gold-standard verification metric.
        master_thread_count = 0
        try:
            master_thread_count = len(_os.listdir(f"/proc/{_os.getpid()}/task"))
        except Exception:
            master_thread_count = -1

        # Probe-client cache stats (lazy import to avoid cyclic import at
        # module load; tools.run_custom_eval imports _scheduler).
        probe_stats: dict = {}
        try:
            from tools.run_custom_eval import get_probe_client_cache_stats
            probe_stats = get_probe_client_cache_stats()
        except Exception:
            pass

        with self._lock:
            tracked = len(self._worker_threads)
            in_flight = len(self._in_flight)
            reaped_total = self._reaper_total_reaped
            reaped_calls = self._reaper_call_count

        self._append_decision_log({
            "event": "scheduler_health",
            "ts_monotonic": now,
            "master_thread_count": master_thread_count,
            "worker_threads_tracked": tracked,
            "in_flight": in_flight,
            "reaper_total_reaped": reaped_total,
            "reaper_call_count": reaped_calls,
            "probe_cache_hits": probe_stats.get("hits", 0),
            "probe_cache_misses": probe_stats.get("misses", 0),
            "probe_cache_evictions": probe_stats.get("evictions", 0),
            "probe_cache_size": probe_stats.get("cached_clients", 0),
        })

        # Also print to stdout so the operator sees it without grepping the
        # tick log. Mirrors the [POOL_STUCK_KILLER_HEARTBEAT] format the user
        # is already accustomed to scanning for.
        try:
            print(
                f"[SCHEDULER_HEALTH] master_threads={master_thread_count} "
                f"workers_tracked={tracked} in_flight={in_flight} "
                f"reaped_total={reaped_total} "
                f"probe_cache=hits/{probe_stats.get('hits',0)}"
                f"/miss/{probe_stats.get('misses',0)}/evict/{probe_stats.get('evictions',0)}"
                f"/size/{probe_stats.get('cached_clients',0)}",
                flush=True,
            )
        except Exception:
            pass

    def run(self) -> None:
        """Block until every sub-queue is empty and no jobs are in flight.

        The run loop is single-threaded at the *decision* level (pick a
        job, pick a GPU, acquire a lease), but spawns one worker thread
        per dispatched job to overlap grandchild execution.  That's what
        produces the throughput win.
        """
        # Startup banner so the operator knows the thread-leak patches are
        # active. Look for [SCHEDULER_HEALTH] lines every ~30s after this
        # to confirm master_threads stays bounded.
        try:
            initial_threads = len(_os.listdir(f"/proc/{_os.getpid()}/task"))
        except Exception:
            initial_threads = -1
        print(
            f"[SCHEDULER_INIT] thread-leak patches ACTIVE: worker-thread reaper + "
            f"carla.Client probe cache. master_threads_at_start={initial_threads}; "
            f"watch for [SCHEDULER_HEALTH] every {self._health_emit_interval_s:.0f}s.",
            flush=True,
        )
        while True:
            # Reap completed workers BEFORE deciding whether to break or
            # block; otherwise a long-running pool slowly leaks OS task
            # entries until the master hits the per-process thread cap and
            # crashes.
            self._reap_dead_workers()
            # Periodic health emission (~every 30s; throttled internally).
            self._emit_scheduler_health()
            with self._cond:
                if self._shutdown:
                    break
                # Drain mode: stop accepting new dispatches.  Exit only
                # after all in-flight finish.
                if self._drain_mode:
                    if not self._in_flight:
                        break
                    self._cond.wait(timeout=self._dispatch_tick_s)
                    continue
                # Done if queues are empty AND nothing in flight AND no
                # external hold is active (e.g. CoLMDriver phase transition
                # waiting on vLLM startup before enqueueing deferred jobs).
                if (
                    not self._in_flight
                    and all(not q.jobs for q in self._queues.values())
                    and self._external_holds <= 0
                ):
                    break

                # Smart dispatch: try the WFQ-preferred job first; if it
                # doesn't fit, look for ANY queued job that does fit.  This
                # prevents the "WFQ keeps picking lmdrive but only 7 GB free
                # so nothing dispatches even though tcp would fit" failure
                # mode that strands free VRAM on partially-loaded GPUs.
                job = self._next_job_weighted_rr()
                if job is None:
                    self._cond.wait(timeout=self._dispatch_tick_s)
                    continue

                placement = self._pick_gpu_and_decide_carla(job)
                if placement is None:
                    # WFQ-preferred job doesn't fit anywhere.  Put it back
                    # at head of its queue, then walk ALL other queues
                    # looking for a smaller job that fits.  Limited to a
                    # bounded scan so dispatch latency stays low.
                    self._queues[job.planner_name].jobs.insert(0, job)
                    deferred_jid = job.job_id
                    deferred_planner = job.planner_name
                    deferred_cost = int(job.vram_estimate_mib)
                    fallback_job = None
                    fallback_placement = None
                    # Sort sub-queues by their FIRST job's planner_cost ASC
                    # (smallest first) so we try cheapest jobs first.
                    candidates = []
                    for qname, q in self._queues.items():
                        if not q.jobs: continue
                        if qname == deferred_planner: continue  # already tried
                        candidates.append((int(q.jobs[0].vram_estimate_mib), qname, q))
                    candidates.sort(key=lambda x: x[0])
                    for _, qname, q in candidates[:8]:  # bounded scan: 8 queues max
                        if not q.jobs: continue
                        candidate_job = q.jobs[0]
                        candidate_placement = self._pick_gpu_and_decide_carla(candidate_job)
                        if candidate_placement is not None:
                            # Found a fitting job!  Pop it and use it.
                            q.jobs.pop(0)
                            q.vtime += 1.0 / max(q.weight, 1e-6)
                            fallback_job = candidate_job
                            fallback_placement = candidate_placement
                            break
                    if fallback_job is not None:
                        job = fallback_job
                        placement = fallback_placement
                        self._append_decision_log({
                            "event": "smart_dispatch_fallback",
                            "ts_monotonic": time.monotonic(),
                            "deferred_job_id": deferred_jid,
                            "deferred_planner": deferred_planner,
                            "deferred_planner_cost_mib": deferred_cost,
                            "dispatched_job_id": job.job_id,
                            "dispatched_planner": job.planner_name,
                            "dispatched_planner_cost_mib": int(job.vram_estimate_mib),
                        })
                        # Fall through to dispatch logic below
                    else:
                        # Truly nothing fits.  Log defer and wait.
                        self._append_decision_log({
                            "event": "defer_no_capacity",
                            "ts_monotonic": time.monotonic(),
                            "job_id": job.job_id,
                            "planner": job.planner_name,
                            "scenario": job.scenario_name,
                            "town": job.town,
                            "planner_vram_estimate_mib": int(job.vram_estimate_mib),
                            "carla_vram_estimate_mib": int(self._carla_vram_estimate_mib),
                            "vram_safety_buffer_mib": int(self._vram_safety_buffer_mib),
                            "retry_attempt": int(job.retry_attempt),
                            "retry_budget": int(job.retry_budget),
                            "free_requeues": int(getattr(job, "free_requeues", 0)),
                            "in_flight": len(self._in_flight),
                            "queued_total": sum(
                                len(q.jobs) for q in self._queues.values()
                            ),
                            "gpus": self._snapshot_gpu_state(),
                        })
                        self._cond.wait(timeout=self._dispatch_tick_s)
                        continue

                gpu_id, need_new_carla = placement
                self._append_decision_log({
                    "event": "dispatch",
                    "ts_monotonic": time.monotonic(),
                    "job_id": job.job_id,
                    "planner": job.planner_name,
                    "scenario": job.scenario_name,
                    "town": job.town,
                    "chosen_gpu": int(gpu_id),
                    "need_new_carla": bool(need_new_carla),
                    "planner_vram_estimate_mib": int(job.vram_estimate_mib),
                    "carla_vram_estimate_mib": int(self._carla_vram_estimate_mib),
                    "retry_attempt": int(job.retry_attempt),
                    "retry_budget": int(job.retry_budget),
                    "free_requeues": int(getattr(job, "free_requeues", 0)),
                    "in_flight_before": len(self._in_flight),
                    "gpus": self._snapshot_gpu_state(),
                })

                # Resolve / create the CARLA instance under the scheduler
                # lock so another concurrent dispatcher couldn't race us.
                # NB: start_instance itself takes significant time and
                # holds the pool's lock, not ours -- we drop our lock
                # before calling it.
            # --- out of lock ---

            if need_new_carla:
                instance = self._pool.start_instance(gpu_id=gpu_id)
                if instance is None:
                    with self._lock:
                        n = self._gpu_consecutive_start_failures.get(gpu_id, 0) + 1
                        self._gpu_consecutive_start_failures[gpu_id] = n
                        if n >= self._gpu_start_failure_threshold:
                            doublings = (n - self._gpu_start_failure_threshold) // 3
                            backoff_s = min(
                                self._gpu_start_max_backoff_s,
                                self._gpu_start_base_backoff_s * (2 ** doublings),
                            )
                            self._gpu_start_blocked_until[gpu_id] = (
                                time.monotonic() + backoff_s
                            )
                            log.warning(
                                "scheduler: gpu=%d blacklisted for %.0fs "
                                "after %d consecutive start failures",
                                gpu_id,
                                backoff_s,
                                n,
                            )
                        else:
                            log.warning(
                                "scheduler: start_instance failed on gpu=%d; "
                                "consecutive_failures=%d; requeueing job=%s",
                                gpu_id,
                                n,
                                job.job_id,
                            )
                    with self._cond:
                        self._queues[job.planner_name].jobs.insert(0, job)
                        self._cond.wait(timeout=self._dispatch_tick_s)
                    continue
                # Successful start — reset failure counter for this GPU.
                with self._lock:
                    self._gpu_consecutive_start_failures[gpu_id] = 0
                    self._gpu_start_blocked_until[gpu_id] = 0.0
                instance_id = instance.instance_id
            else:
                # Reuse path: find a CLEAN instance matching town on this GPU.
                chosen = self._pool.find_available_instance(town=job.town, gpu_id=gpu_id)
                if chosen is None:
                    # Warm instance was stolen by another dispatch; retry.
                    with self._cond:
                        self._queues[job.planner_name].jobs.insert(0, job)
                        self._cond.wait(timeout=self._dispatch_tick_s)
                    continue
                instance_id = chosen.instance_id

            lease = self._pool.acquire_lease(instance_id, town=job.town)
            if lease is None:
                log.warning(
                    "scheduler: acquire_lease failed instance=%s; requeueing job=%s",
                    instance_id,
                    job.job_id,
                )
                with self._cond:
                    self._queues[job.planner_name].jobs.insert(0, job)
                    self._cond.wait(timeout=self._dispatch_tick_s)
                continue

            # Reserve VRAM and dispatch the grandchild asynchronously.
            with self._lock:
                self._reserve_gpu_vram(
                    gpu_id,
                    self._vram_cost_for_job(job, need_new_carla=need_new_carla),
                )
                self._in_flight[job.job_id] = job

            t = threading.Thread(
                target=self._run_one_job,
                args=(job, lease, gpu_id, need_new_carla),
                name=f"scn-{job.job_id}",
                daemon=True,
            )
            self._worker_threads.append(t)
            t.start()

        # Drain: wait for in-flight threads to finish.
        for t in list(self._worker_threads):
            t.join(timeout=self._dispatch_tick_s)

    def shutdown(self) -> None:
        """Signal the run loop to drain in-flight work and exit."""
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()

    def acquire_external_hold(self) -> None:
        """Block run() from exiting due to empty-queue/empty-flight even
        when both are zero.  The caller is responsible for matching every
        acquire with a release once the external work (e.g. starting vLLM
        and enqueueing deferred jobs) is complete."""
        with self._cond:
            self._external_holds += 1

    def release_external_hold(self) -> None:
        with self._cond:
            self._external_holds = max(0, self._external_holds - 1)
            self._cond.notify_all()

    def enter_drain_mode(self) -> None:
        """Stop accepting new dispatches.  Let in-flight jobs finish naturally.
        Called by the SIGUSR1 handler so the operator can apply config changes
        without losing in-flight compute.  Use ``shutdown()`` for hard exit."""
        with self._cond:
            self._drain_mode = True
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Per-job worker
    # ------------------------------------------------------------------

    def _run_one_job(
        self,
        job: ScenarioJob,
        lease,                         # CarlaLease
        gpu_id: int,
        need_new_carla: bool,
    ) -> None:
        """Spawn the grandchild, wait for it, release the lease, record result."""
        start = time.monotonic()
        result: Optional[ScenarioJobResult] = None
        try:
            result = self._spawn_grandchild(job=job, lease=lease)
        except Exception as exc:
            log.exception("scheduler: spawn_grandchild raised for job=%s: %s", job.job_id, exc)
            result = ScenarioJobResult(
                job=job,
                status=JobStatus.SOFT_FAILURE,
                result_root=Path(job.results_tag) / job.scenario_name,
                duration_s=time.monotonic() - start,
                lease_gpu_id=gpu_id,
                carla_instance_id=lease.instance_id,
                carla_generation=lease.generation,
                carla_release_status="crashed",
                stderr_tail=str(exc),
            )
        finally:
            try:
                self._pool.release_lease(
                    lease,
                    status=(result.carla_release_status if result is not None else "crashed"),
                )
            except Exception as exc:
                log.warning("scheduler: release_lease raised job=%s: %s", job.job_id, exc)
            # Release VRAM reservation.
            with self._cond:
                self._release_gpu_vram(
                    gpu_id,
                    self._vram_cost_for_job(job, need_new_carla=need_new_carla),
                )
                self._in_flight.pop(job.job_id, None)
                self._cond.notify_all()

            # OOM/success backoff feedback.
            if result is not None:
                if result.status == JobStatus.OOM:
                    self.record_oom(gpu_id)
                elif result.status == JobStatus.SUCCESS:
                    self.record_success(gpu_id)

            # Always notify the observer so its counters (running / failed)
            # stay consistent.  The reporter is responsible for treating
            # ``retry_queued`` differently from a terminal failure.
            if result is not None:
                try:
                    self._on_result(result)
                except Exception as exc:
                    log.exception(
                        "scheduler: on_result raised job=%s: %s",
                        job.job_id,
                        exc,
                    )
                try:
                    self._append_decision_log({
                        "event": "job_complete",
                        "ts_monotonic": time.monotonic(),
                        "job_id": job.job_id,
                        "planner": job.planner_name,
                        "scenario": job.scenario_name,
                        "status": result.status.value,
                        "duration_s": round(float(result.duration_s), 3),
                        "gpu_id": int(gpu_id),
                        "carla_instance_id": result.carla_instance_id,
                        "carla_release_status": result.carla_release_status,
                        "free_requeue": bool(getattr(result, "free_requeue", False)),
                        "retry_attempt": int(job.retry_attempt),
                        "retry_budget": int(job.retry_budget),
                        "free_requeues": int(getattr(job, "free_requeues", 0)),
                        "in_flight_after": len(self._in_flight),
                    })
                except Exception:
                    log.exception(
                        "scheduler: decision-log append on job_complete failed "
                        "job=%s",
                        job.job_id,
                    )

            # Retry re-enqueue happens AFTER on_result so the reporter can
            # snap its counters before the next dispatch fires.
            if result is not None and result.status == JobStatus.RETRY_QUEUED:
                if getattr(result, "free_requeue", False):
                    # Infra failure (CARLA unhealthy before dispatch etc.) --
                    # do not charge this against the job's retry budget.
                    # But cap the number of free requeues so a permanently
                    # unhealthy GPU can't trap a single job here forever.
                    if job.free_requeues >= self._max_free_requeues_per_job:
                        log.warning(
                            "scheduler: job=%s exhausted free-requeue cap "
                            "(%d); downgrading to regular retry",
                            job.job_id,
                            self._max_free_requeues_per_job,
                        )
                        retry_job = dataclasses.replace(
                            job,
                            retry_attempt=job.retry_attempt + 1,
                        )
                    else:
                        retry_job = dataclasses.replace(
                            job,
                            free_requeues=job.free_requeues + 1,
                        )
                else:
                    retry_job = dataclasses.replace(
                        job, retry_attempt=job.retry_attempt + 1
                    )
                self.enqueue_front(retry_job)

    # ------------------------------------------------------------------
    # OOM feedback
    # ------------------------------------------------------------------

    def record_oom(self, gpu_id: int) -> None:
        """Reduce ``gpu_id``'s effective VRAM budget by ``oom_backoff_factor``."""
        with self._lock:
            prev = self._gpu_oom_multiplier.get(gpu_id, 1.0)
            self._gpu_oom_multiplier[gpu_id] = max(
                self._oom_multiplier_floor, prev * self._oom_backoff_factor
            )
            self._gpu_successes_since_oom[gpu_id] = 0
            self._gpu_last_oom_monotonic[gpu_id] = time.monotonic()
            log.warning(
                "scheduler: OOM on gpu=%d, multiplier %.2f -> %.2f",
                gpu_id,
                prev,
                self._gpu_oom_multiplier[gpu_id],
            )

    def _maybe_recover_oom_multiplier_idle(self, gpu_id: int) -> None:
        """Deadlock-recovery: if a knocked-down GPU is idle and has not OOMed
        for a while, gently nudge its multiplier back up so large planners get
        a chance to run again.  Without this, a GPU that OOMed during a burst
        of small-planner traffic stays wedged at the floor forever once those
        small planners drain, because the success-based recovery path
        (``record_success``) needs jobs to actually run -- and nothing fits.
        """
        mult = self._gpu_oom_multiplier.get(gpu_id, 1.0)
        if mult >= 1.0:
            return
        active = int(self._gpu_active_leases.get(gpu_id, 0))
        if active > 0:
            # GPU is actively being probed; defer to success-based recovery.
            return
        last_oom = float(self._gpu_last_oom_monotonic.get(gpu_id, 0.0))
        if last_oom <= 0.0:
            return
        idle_for = time.monotonic() - last_oom
        if idle_for < self._oom_multiplier_recovery_idle_s:
            return
        factor = self._oom_backoff_factor
        if factor <= 0.0 or factor >= 1.0:
            return
        new_mult = min(1.0, mult / factor)
        # Reset the OOM clock so we don't race-recover multiple steps at once.
        self._gpu_last_oom_monotonic[gpu_id] = time.monotonic()
        if new_mult > mult:
            self._gpu_oom_multiplier[gpu_id] = new_mult
            log.warning(
                "scheduler: idle-recovery on gpu=%d, multiplier %.2f -> %.2f "
                "(idle_for=%.0fs, active_leases=%d)",
                gpu_id,
                mult,
                new_mult,
                idle_for,
                active,
            )

    def _append_decision_log(self, payload: dict) -> None:
        """Write one JSON line to ``--scheduler-tick-log`` if configured.

        Called from dispatch decisions.  Best-effort (IO errors are logged
        but never propagate) so decision-log plumbing can never stall the
        scheduler itself.
        """
        if self._decision_log_path is None:
            return
        try:
            import json
            line = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            log.exception("scheduler: could not serialise decision log payload")
            return
        try:
            with self._decision_log_lock:
                with self._decision_log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            log.exception(
                "scheduler: decision log write failed path=%s",
                self._decision_log_path,
            )

    def _snapshot_gpu_state(self) -> dict[int, dict]:
        """Return a summary of per-GPU state for the decision log."""
        out: dict[int, dict] = {}
        for gpu_id in self._gpu_ids:
            stats = self._gpu_monitor.stats_for(str(gpu_id))
            if stats is None:
                out[int(gpu_id)] = {
                    "stats": "stale_or_unavailable",
                    "active_leases": int(self._gpu_active_leases.get(gpu_id, 0)),
                    "oom_multiplier": round(
                        float(self._gpu_oom_multiplier.get(gpu_id, 1.0)), 3
                    ),
                }
                continue
            reserved = int(self._gpu_reserved_mib.get(gpu_id, 0))
            out[int(gpu_id)] = {
                "free_mib": int(getattr(stats, "free_mib", 0)),
                "used_mib": int(getattr(stats, "used_mib", 0)),
                "total_mib": int(getattr(stats, "total_mib", 0)),
                "reserved_mib": reserved,
                "unobserved_reserved_mib": int(
                    self._unobserved_reserved_mib(gpu_id, stats=stats)
                ),
                "active_leases": int(self._gpu_active_leases.get(gpu_id, 0)),
                "oom_multiplier": round(
                    float(self._gpu_oom_multiplier.get(gpu_id, 1.0)), 3
                ),
            }
        return out

    def record_success(self, gpu_id: int) -> None:
        """Recover 1/N of the OOM knockdown per successful scenario on the GPU."""
        with self._lock:
            cur = self._gpu_successes_since_oom.get(gpu_id, 0) + 1
            self._gpu_successes_since_oom[gpu_id] = cur
            if cur >= self._oom_backoff_recover_count:
                mult = self._gpu_oom_multiplier.get(gpu_id, 1.0)
                if mult < 1.0:
                    factor = self._oom_backoff_factor
                    if factor > 0:
                        self._gpu_oom_multiplier[gpu_id] = min(1.0, mult / factor)
                        self._gpu_successes_since_oom[gpu_id] = 0

    # ------------------------------------------------------------------
    # Dispatch decision (pure; unit-testable)
    # ------------------------------------------------------------------

    def _vram_cost_for_job(self, job: ScenarioJob, *, need_new_carla: bool) -> int:
        planner_cost = max(0, int(job.vram_estimate_mib))
        carla_cost = self._carla_vram_estimate_mib if need_new_carla else 0
        return planner_cost + carla_cost

    def _effective_budget_mib(self, gpu_id: int) -> Optional[int]:
        """Current usable VRAM on ``gpu_id`` in MiB, or None if the snapshot
        is stale (the caller should decline to dispatch until nvidia-smi
        catches up).
        """
        if not self._gpu_monitor.has_fresh_snapshot(self._fresh_snapshot_max_age_s):
            return None
        stats = self._gpu_monitor.stats_for(str(gpu_id))
        if stats is None:
            return None
        # Idle-recovery probe: if this GPU is sidelined by an old OOM and has
        # been quiet long enough, gently lift the multiplier so we can try
        # admitting a larger planner again.  Called here (not in record_oom)
        # because it needs to fire even when nothing is running on the GPU.
        self._maybe_recover_oom_multiplier_idle(gpu_id)
        free_mib = int(getattr(stats, "free_mib", 0))
        unobserved_reserved = self._unobserved_reserved_mib(gpu_id, stats=stats)
        raw_budget = max(0, free_mib - self._vram_safety_buffer_mib - unobserved_reserved)
        multiplier = float(self._gpu_oom_multiplier.get(gpu_id, 1.0))
        return int(raw_budget * multiplier)

    def _reserve_gpu_vram(self, gpu_id: int, amount_mib: int) -> None:
        # Always bump the active-lease counter on dispatch, regardless of the
        # VRAM amount.  The concurrency cap in ``_pick_gpu_and_decide_carla``
        # reads this.
        self._gpu_active_leases[gpu_id] = self._gpu_active_leases.get(gpu_id, 0) + 1
        # Record dispatch time so the per-GPU cooldown can serialize
        # consecutive dispatches and avoid the OOM race.
        self._gpu_last_dispatch_at[gpu_id] = time.monotonic()
        amount_mib = max(0, int(amount_mib))
        if amount_mib <= 0:
            return
        self._gpu_reserved_mib[gpu_id] = self._gpu_reserved_mib.get(gpu_id, 0) + amount_mib
        window = self._gpu_reservation_windows.setdefault(gpu_id, _GpuReservationWindow())
        stats = self._gpu_monitor.stats_for(str(gpu_id))
        current_used = None if stats is None else int(getattr(stats, "used_mib", 0))
        if window.baseline_used_mib is None:
            window.baseline_used_mib = current_used
            window.peak_used_mib = current_used
        elif current_used is not None:
            prev_peak = (
                current_used
                if window.peak_used_mib is None
                else max(int(window.peak_used_mib), current_used)
            )
            window.peak_used_mib = prev_peak

    def _release_gpu_vram(self, gpu_id: int, amount_mib: int) -> None:
        # Symmetric to _reserve_gpu_vram: always drop the lease counter even
        # if the amount was 0.
        cur_active = int(self._gpu_active_leases.get(gpu_id, 0))
        self._gpu_active_leases[gpu_id] = max(0, cur_active - 1)
        amount_mib = max(0, int(amount_mib))
        remaining = max(0, int(self._gpu_reserved_mib.get(gpu_id, 0)) - amount_mib)
        self._gpu_reserved_mib[gpu_id] = remaining
        if remaining > 0:
            return
        window = self._gpu_reservation_windows.setdefault(gpu_id, _GpuReservationWindow())
        window.reserved_mib = 0
        window.baseline_used_mib = None
        window.peak_used_mib = None

    def _unobserved_reserved_mib(self, gpu_id: int, *, stats) -> int:
        # MAXIMUM CONCURRENCY MODE: by default, trust nvidia-smi entirely and
        # do NOT hold back any "unobserved" reservation.  Forensics showed
        # this accounting was holding back ~75 GB of VRAM across 8 GPUs --
        # roughly 13 worth of additional concurrent evaluator slots -- because
        # planners typically use less VRAM than their conservative estimates.
        # Set CARLA_TRUST_OBSERVED_VRAM_ONLY=0 to restore the prior behavior.
        import os
        if os.environ.get("CARLA_TRUST_OBSERVED_VRAM_ONLY", "1").lower() in ("1", "true", "yes", "on"):
            return 0

        reserved = int(self._gpu_reserved_mib.get(gpu_id, 0))
        if reserved <= 0:
            return 0

        window = self._gpu_reservation_windows.setdefault(gpu_id, _GpuReservationWindow())
        current_used = int(getattr(stats, "used_mib", 0))
        if window.baseline_used_mib is None:
            window.baseline_used_mib = current_used
            window.peak_used_mib = current_used
            return reserved

        if window.peak_used_mib is None or current_used > int(window.peak_used_mib):
            window.peak_used_mib = current_used

        observed_growth = max(
            0,
            int(window.peak_used_mib or current_used) - int(window.baseline_used_mib),
        )
        return max(0, reserved - observed_growth)

    def _pick_gpu_and_decide_carla(
        self,
        job: ScenarioJob,
    ) -> Optional[tuple[int, bool]]:
        """Return ``(gpu_id, need_new_carla)`` or None if no capacity right now.

        Decision tree:
          1. Is there a CLEAN pool instance with ``job.town`` already loaded?
             Its GPU just needs ``job.vram_estimate_mib`` of free VRAM.
             Pick among matching warm instances by largest headroom.
             ``need_new_carla = False``.
          2. Is there a CLEAN pool instance with any other town?  Same as
             (1) but will cost a ``load_world`` on acquire; still no new
             CARLA needed.  ``need_new_carla = False``.
          3. Otherwise, pick the GPU with enough headroom for
             ``planner_vram + carla_vram`` and start a new CARLA there.
          4. Otherwise, None.
        """
        planner_cost = max(0, int(job.vram_estimate_mib))
        carla_cost = self._carla_vram_estimate_mib

        # Build an ordered list of (gpu_id, effective_budget_mib) with
        # stale-snapshot GPUs excluded entirely.  Also exclude GPUs that are
        # already at the per-GPU concurrency cap -- we can't overschedule
        # compute even when VRAM still fits, because the CARLA server's
        # own tick loop will starve under too many concurrent tenants and
        # trip the heartbeat watchdog (observed as reason=stalled / rc=-9).
        gpu_budgets: list[tuple[int, int]] = []
        now = time.monotonic()
        for gpu in self._gpu_ids:
            if self._max_leases_per_gpu is not None:
                active = int(self._gpu_active_leases.get(gpu, 0))
                if active >= self._max_leases_per_gpu:
                    continue
            # Per-GPU dispatch cooldown: skip GPUs that just received a
            # dispatch within the cooldown window.  Prevents the OOM race
            # where two planners load on the same GPU before nvidia-smi
            # reflects the first one's allocation.
            if self._gpu_dispatch_cooldown_s > 0:
                last_disp = self._gpu_last_dispatch_at.get(gpu, 0.0)
                if last_disp > 0 and (now - last_disp) < self._gpu_dispatch_cooldown_s:
                    continue
            b = self._effective_budget_mib(gpu)
            if b is not None:
                gpu_budgets.append((gpu, b))
        if not gpu_budgets:
            return None

        # --- Priority 1: same-town warm instance ---
        for gpu_id, budget in sorted(gpu_budgets, key=lambda x: -x[1]):
            inst = self._pool.find_available_instance(town=job.town, gpu_id=gpu_id)
            if inst is None or inst.current_town != job.town:
                continue
            if budget >= planner_cost:
                return (gpu_id, False)

        # --- Priority 2: any-town warm instance on any GPU ---
        for gpu_id, budget in sorted(gpu_budgets, key=lambda x: -x[1]):
            inst = self._pool.find_available_instance(gpu_id=gpu_id)
            if inst is None:
                continue
            if budget >= planner_cost:
                return (gpu_id, False)

        # --- Priority 3: start a new CARLA on a GPU with enough headroom ---
        # Skip GPUs that are temporarily blacklisted due to consecutive
        # start failures (the backoff expires automatically).
        now = time.monotonic()
        for gpu_id, budget in sorted(gpu_budgets, key=lambda x: -x[1]):
            if now < self._gpu_start_blocked_until.get(gpu_id, 0.0):
                continue
            if budget >= planner_cost + carla_cost:
                return (gpu_id, True)

        return None

    def _next_job_weighted_rr(self) -> Optional[ScenarioJob]:
        """Weighted fair queueing pick across per-planner sub-queues.

        The sub-queue with the smallest ``vtime`` among those with jobs is
        selected; after we take a job, its ``vtime`` advances by
        ``1/weight``.  Over time, queues with higher weight are sampled
        more frequently, so fast planners ship more often.
        """
        eligible = [q for q in self._queues.values() if q.jobs]
        if not eligible:
            return None
        q = min(eligible, key=lambda x: x.vtime)
        job = q.jobs.pop(0)
        q.vtime += 1.0 / q.weight
        return job


# ----------------------------------------------------------------------
# Helpers exposed to run_custom_eval for wiring
# ----------------------------------------------------------------------


def compute_planner_weight(mean_run_scenario_s: float, floor: float = 60.0) -> float:
    """Weight = 1 / max(mean_run_scenario_s, floor).

    Faster planners get higher weight and more frequent dispatch under
    weighted-RR.  The ``floor`` protects us from a single very-short
    scenario blowing up the weight.
    """
    return 1.0 / max(float(mean_run_scenario_s), floor)
