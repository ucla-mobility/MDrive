"""CARLA process pool for the scenario-pool scheduler.

This module defines a pool of long-lived CARLA server processes that are
leased out to grandchild evaluator processes one scenario at a time.

Key properties
--------------
* VRAM accounting is OWNED BY THE SCHEDULER, not the pool.  The pool only
  manages process lifecycle -- the scheduler decides when it is safe to start
  a new CARLA instance on a given GPU by consulting its own VRAM model and
  OOM penalty state.  When the scheduler has decided, it calls
  ``CarlaPool.start_instance(gpu_id, ...)`` which picks a free port from the
  GPU's candidate band and brings the process up.
* Leases are exclusive: one scenario holds a given CARLA at a time.  A
  concurrently running scenario gets its own lease on a different CARLA
  instance.  There is no notion of concurrent scenarios sharing one CARLA.
* Port selection is verified free before bind: each GPU has a *candidate*
  port band (e.g. ``2000 + gpu_id*20 + [0,4,8,12,16]``); the pool iterates
  the band and checks ``are_ports_available`` before binding.
* Recycle-on-crash only.  A successful ``release(lease, "clean")`` keeps the
  CARLA instance warm for the next lease (town preferred); there is no
  prophylactic K-scenario recycle.  A ``release(lease, "crashed")`` triggers
  kill + wait_for_port_close.  Construct the pool with ``no_reuse=True``
  to fall back to always-recycle behavior (useful for debugging /
  scenario-isolation).

This module is the process-lifecycle half of the scenario-pool redesign.
The scheduling/dispatch half lives in :mod:`tools._scheduler`.
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional
try:
    from typing import Literal  # Python 3.8+
except ImportError:
    from typing_extensions import Literal  # Python 3.7 fallback (colmdrivermarco2)


log = logging.getLogger(__name__)


# Module-global handle to the live pool so EmergencyGuard can read its
# bookkeeping without explicit plumbing.  Set by ``CarlaPool.__init__``;
# only one pool exists per master process.
_ACTIVE_POOL: "CarlaPool | None" = None


def get_active_pool() -> "CarlaPool | None":
    """Return the live ``CarlaPool`` instance for this process, or None."""
    return _ACTIVE_POOL


class InstanceState(str, enum.Enum):
    """Lifecycle states of a pooled CARLA instance."""

    STARTING = "starting"        # subprocess spawned, not yet RPC-ready
    IDLE = "idle"                # RPC-ready, no town loaded, no lease
    CLEAN = "clean"              # has a town loaded, no lease, safe to hand out
    LEASED = "leased"            # held by one scenario
    RESETTING = "resetting"      # post-release defensive cleanup in progress
    RECYCLING = "recycling"      # killed or killing; port closing
    DEAD = "dead"                # terminal; will be removed from the pool


@dataclasses.dataclass
class CarlaLease:
    """An exclusive loan of one CARLA instance to one scenario."""

    lease_id: str
    instance_id: str
    gpu_id: int
    host: str
    world_port: int
    tm_port: int
    current_town: Optional[str]
    generation: int
    acquired_at: float


@dataclasses.dataclass
class PoolInstance:
    """Pool-internal bookkeeping for a single CARLA process."""

    instance_id: str
    gpu_id: int
    host: str
    world_port: int
    tm_port: int
    manager: Any                 # CarlaProcessManager (lazy import)
    state: InstanceState = InstanceState.STARTING
    current_town: Optional[str] = None
    generation: int = 0
    scenarios_served: int = 0
    created_at: float = dataclasses.field(default_factory=time.monotonic)
    last_release_status: Optional[str] = None
    # Monotonic timestamp of the most recent lease release.  Used by the
    # idle-recycle reaper: any instance in state CLEAN whose
    # ``last_released_at`` is more than ``idle_recycle_seconds`` ago is
    # eligible to be stopped to free VRAM.  Unset when the instance is
    # freshly started (``STARTING`` / ``IDLE``) because those transitions
    # aren't lease releases.
    last_released_at: Optional[float] = None
    # True while a lease is out on this instance.
    _lease_id: Optional[str] = None


# Status a lease returns with, driving the pool's post-release action.
LeaseStatus = Literal["clean", "dirty", "crashed"]


# ----------------------------------------------------------------------
# Lazy imports
# ----------------------------------------------------------------------
# run_custom_eval imports this module, so we can't import it at module
# load time without creating a cycle.  All references to CarlaProcessManager,
# port helpers, and find_carla_egg happen inside methods.


def _rce():
    """Return the run_custom_eval module, imported lazily."""
    import tools.run_custom_eval as _m  # noqa: WPS433 -- deliberate lazy import
    return _m


# ----------------------------------------------------------------------


class CarlaPool:
    """Dynamic pool of long-lived CARLA processes.

    The pool does NOT cap itself by slot count.  Capacity is emergent from
    what the scheduler asks for (``start_instance``) and what it releases
    (``stop_instance``).  When the scheduler observes low VRAM or OOM
    history, it simply stops asking for new instances; when headroom
    recovers, it asks again.
    """

    def __init__(
        self,
        *,
        carla_root: Path,
        host: str = "127.0.0.1",
        base_env: Optional[dict] = None,
        extra_args: Optional[List[str]] = None,
        port_band_fn: Callable[[int], Iterable[tuple[int, int]]] | None = None,
        tm_port_offset: int = 2,
        restart_wait_s: float = 8.0,
        post_start_buffer_s: float = 5.0,
        rpc_ready_timeout_s: float = 60.0,
        port_close_timeout_s: float = 30.0,
        no_reuse: bool = False,
        quiet: bool = True,
        status_callback: Callable[[str, dict], None] | None = None,
        idle_recycle_seconds: float = 180.0,
        idle_recycle_poll_seconds: float = 30.0,
    ) -> None:
        """Create an empty pool.

        Args:
            carla_root: CARLA installation root (containing ``CarlaUE4.sh``).
            host: Host CARLA listens on (usually localhost).
            base_env: Environment passed to CARLA (minus ``CUDA_VISIBLE_DEVICES``
                which the pool sets per-instance).
            extra_args: Extra arguments forwarded to CarlaUE4.sh (e.g. quality).
            port_band_fn: Called as ``port_band_fn(gpu_id)`` -> iterable of
                ``(world_port, tm_port)`` candidates for that GPU.  Defaults to
                :func:`default_port_band`.
            tm_port_offset: Offset from world_port to tm_port (must match CARLA
                convention; 2 by default).
            restart_wait_s: Sleep after CARLA stop before restart.
            post_start_buffer_s: Sleep after ``start`` returns before the
                instance is considered usable.
            rpc_ready_timeout_s: Max time to wait for CARLA RPC to respond.
            port_close_timeout_s: Max time to wait for a released port to
                return to CLOSED/unbound state before reuse.
            no_reuse: If True, every release -- regardless of status --
                triggers ``stop_instance``.  This disables CARLA reuse
                entirely (each job gets a fresh instance).  Used to isolate
                scenarios when debugging or until Phase 4 reuse policies
                are validated.
        """
        self._carla_root = Path(carla_root)
        self._host = host
        self._base_env = dict(base_env or {})
        self._extra_args = list(extra_args or [])
        self._port_band_fn = port_band_fn or default_port_band
        self._tm_port_offset = int(tm_port_offset)
        self._restart_wait_s = float(restart_wait_s)
        self._post_start_buffer_s = float(post_start_buffer_s)
        self._rpc_ready_timeout_s = float(rpc_ready_timeout_s)
        self._port_close_timeout_s = float(port_close_timeout_s)
        self._no_reuse = bool(no_reuse)
        self._quiet = bool(quiet)
        self._status_callback = status_callback

        self._lock = threading.RLock()
        self._instances: dict[str, PoolInstance] = {}
        self._leases: dict[str, CarlaLease] = {}
        self._instance_seq_per_gpu: dict[int, itertools.count] = {}
        # Set by EmergencyGuard immediately before its kill phase: blocks new
        # ``start_instance`` calls so the main loop can't spawn fresh CARLAs
        # while we're tearing down the old ones (which would otherwise cause
        # VRAM to ascend during the drain wait).
        self._starts_paused = False

        # Register globally so the EmergencyGuard can find this pool without
        # explicit plumbing.  Last-pool-wins is fine: there is only ever one
        # CarlaPool per master process.
        global _ACTIVE_POOL
        _ACTIVE_POOL = self

        # Idle-recycle reaper: background thread that stops warm-but-
        # unused CARLAs so they don't hoard VRAM.  Disabled when
        # ``idle_recycle_seconds <= 0``.  See ``_idle_recycle_loop`` for
        # the full rationale.
        self._idle_recycle_seconds = float(max(0.0, idle_recycle_seconds))
        self._idle_recycle_poll_seconds = float(max(5.0, idle_recycle_poll_seconds))
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        if self._idle_recycle_seconds > 0.0:
            self._reaper_thread = threading.Thread(
                target=self._idle_recycle_loop,
                name="CarlaPoolIdleReaper",
                daemon=True,
            )
            self._reaper_thread.start()
            log.info(
                "carla_pool: idle-recycle reaper started "
                "(idle_threshold=%.0fs, poll=%.0fs)",
                self._idle_recycle_seconds,
                self._idle_recycle_poll_seconds,
            )

    # ------------------------------------------------------------------
    # Public: scheduler-driven lifecycle
    # ------------------------------------------------------------------

    def _idle_recycle_loop(self) -> None:
        """Periodically stop CLEAN instances that have been idle too long.

        Rationale: CARLA reuse is great when a new scenario of the same
        town is queued right behind the previous one -- the instance
        stays CLEAN, its ~4 GB of VRAM is already allocated, and
        ``acquire_lease`` hands it back in milliseconds.  But if the
        queue drains of that town's scenarios (weighted-RR picked a
        different planner, the remaining towns are all different), the
        CLEAN instance sits there holding VRAM forever.  On a 30-CARLA
        baseline this typically means ~27 idle CLEANs holding ~100 GB
        across the fleet -- precisely the VRAM the scheduler needed to
        start fresh CARLAs elsewhere.

        We only stop instances that have been CLEAN for the full
        ``idle_recycle_seconds`` window and that have never held a
        lease in progress.  We never touch STARTING / LEASED / IDLE /
        RESETTING / RECYCLING / DEAD instances.
        """
        while not self._reaper_stop.wait(self._idle_recycle_poll_seconds):
            try:
                now = time.monotonic()
                to_stop: list[str] = []
                with self._lock:
                    for inst in self._instances.values():
                        if inst.state != InstanceState.CLEAN:
                            continue
                        released_at = inst.last_released_at
                        if released_at is None:
                            continue
                        idle_for = now - released_at
                        if idle_for >= self._idle_recycle_seconds:
                            to_stop.append(inst.instance_id)
                for instance_id in to_stop:
                    try:
                        log.info(
                            "carla_pool: idle-recycle stopping instance=%s "
                            "(idle > %.0fs)",
                            instance_id,
                            self._idle_recycle_seconds,
                        )
                        self._emit(
                            "carla_idle_recycle",
                            instance_id=instance_id,
                            idle_threshold_s=self._idle_recycle_seconds,
                        )
                        self.stop_instance(instance_id)
                    except Exception:
                        log.exception(
                            "carla_pool: idle-recycle stop failed for %s",
                            instance_id,
                        )
            except Exception:
                log.exception("carla_pool: idle-recycle loop iteration raised")

    def _emit(self, event: str, **payload: Any) -> None:
        cb = self._status_callback
        if cb is None:
            return
        try:
            cb(event, dict(payload))
        except Exception:
            log.exception("carla_pool: status_callback raised (event=%s)", event)

    def start_instance(self, gpu_id: int) -> Optional[PoolInstance]:
        """Spawn a new CARLA process on ``gpu_id``.

        Walks the candidate port band looking for a pair whose ports are
        confirmed free (via ``are_ports_available``), then delegates to
        ``CarlaProcessManager.start()`` to bring up CARLA.  Returns the
        ``PoolInstance`` or ``None`` if no port bundle is free or CARLA
        failed to start.
        """
        rce = _rce()
        with self._lock:
            paused = self._starts_paused
        if paused:
            log.warning(
                "carla_pool: start_instance refused (gpu=%d) — emergency "
                "restart in progress; pool starts paused.",
                gpu_id,
            )
            self._emit("carla_start_failed", gpu_id=gpu_id, reason="starts_paused")
            return None
        port_bundle = self._find_free_port(gpu_id)
        if port_bundle is None:
            log.warning("carla_pool: no free port in band for gpu=%d", gpu_id)
            self._emit("carla_start_failed", gpu_id=gpu_id, reason="no_free_port")
            return None
        world_port, tm_port = port_bundle
        _start_begin_t = time.monotonic()
        self._emit("carla_start_begin", gpu_id=gpu_id, world_port=world_port, tm_port=tm_port)

        env = dict(self._base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        manager = rce.CarlaProcessManager(
            carla_root=self._carla_root,
            host=self._host,
            port=world_port,
            tm_port_offset=self._tm_port_offset,
            extra_args=list(self._extra_args),
            env=env,
            port_tries=1,             # pool already chose a free port
            port_step=4,
            server_log_path=None,
            rootcause_config=None,
            quiet=self._quiet,
        )

        with self._lock:
            seq = next(self._instance_seq_per_gpu.setdefault(gpu_id, itertools.count(1)))
            instance_id = mint_instance_id(gpu_id, seq)
            inst = PoolInstance(
                instance_id=instance_id,
                gpu_id=gpu_id,
                host=self._host,
                world_port=world_port,
                tm_port=tm_port,
                manager=manager,
                state=InstanceState.STARTING,
            )
            self._instances[instance_id] = inst

        try:
            manager.start()
            if self._post_start_buffer_s > 0:
                rce.wait_for_carla_post_start_buffer(self._post_start_buffer_s)
        except Exception as exc:
            log.error(
                "carla_pool: start failed gpu=%d port=%d: %s",
                gpu_id,
                world_port,
                exc,
            )
            with self._lock:
                inst.state = InstanceState.DEAD
                self._instances.pop(instance_id, None)
            # Best-effort cleanup in case CARLA partially started.
            try:
                manager.stop()
            except Exception:
                pass
            try:
                rce.wait_for_port_close(self._host, world_port, timeout=self._port_close_timeout_s)
            except Exception:
                pass
            self._emit(
                "carla_start_failed",
                gpu_id=gpu_id,
                world_port=world_port,
                tm_port=tm_port,
                reason=str(exc),
                duration_s=time.monotonic() - _start_begin_t,
            )
            return None

        with self._lock:
            inst.state = InstanceState.IDLE
        log.info(
            "carla_pool: started instance=%s gpu=%d world_port=%d tm_port=%d",
            instance_id,
            gpu_id,
            world_port,
            tm_port,
        )
        self._emit(
            "carla_start_ready",
            instance_id=instance_id,
            gpu_id=gpu_id,
            world_port=world_port,
            tm_port=tm_port,
            duration_s=time.monotonic() - _start_begin_t,
        )
        return inst

    def stop_instance(self, instance_id: str) -> None:
        """Kill an instance and remove it from the pool.

        Blocks until the process is gone and its ports are CLOSED.  Rejects
        instances currently under lease.
        """
        rce = _rce()
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                return
            if inst.state == InstanceState.LEASED:
                raise RuntimeError(
                    f"carla_pool: cannot stop leased instance {instance_id}"
                )
            inst.state = InstanceState.RECYCLING

        try:
            inst.manager.stop(timeout=15.0)
        except Exception as exc:
            log.warning("carla_pool: stop raised on %s: %s", instance_id, exc)
        try:
            rce.wait_for_port_close(
                self._host,
                inst.world_port,
                timeout=self._port_close_timeout_s,
            )
        except Exception:
            pass

        # Evict any cached probe client for this CARLA's (host, port). If we
        # leave it cached, a future fresh CARLA spawned on the same port will
        # talk to a stale rpclib connection and the probe will spin until
        # timeout. Plus we don't need the client object hanging around -- the
        # underlying CARLA process is gone.
        try:
            from tools.run_custom_eval import _drop_probe_client
            _drop_probe_client(self._host, inst.world_port)
        except Exception:
            pass

        with self._lock:
            inst.state = InstanceState.DEAD
            self._instances.pop(instance_id, None)
        log.info("carla_pool: stopped instance=%s", instance_id)
        self._emit(
            "carla_stopped",
            instance_id=instance_id,
            gpu_id=inst.gpu_id,
            world_port=inst.world_port,
        )

    def list_instances(self) -> list[PoolInstance]:
        """Snapshot of all pool instances (ordered by creation time)."""
        with self._lock:
            return sorted(self._instances.values(), key=lambda p: p.created_at)

    def count_alive_instances(self) -> int:
        """Return the number of pool instances NOT in DEAD state.

        Source of truth for "how many CARLAs are running."  /proc-walking
        is unreliable because CarlaUE4.sh's GDB-detach RPC patch reparents
        the binary to init at birth, so descendant-based counts can return
        0 even when many CARLAs are live.
        """
        with self._lock:
            return sum(
                1 for inst in self._instances.values()
                if inst.state != InstanceState.DEAD
            )

    def live_instance_pids(self) -> list[tuple[str, int]]:
        """Return ``[(instance_id, wrapper_pid), ...]`` for non-DEAD instances.

        ``wrapper_pid`` is ``manager.process.pid`` — the PID of the
        ``CarlaUE4.sh`` bash wrapper.  The actual UE4 binary may be a
        descendant (direct-launch path) or reparented to init (GDB-detach
        path); callers that need to kill the binary should additionally
        sweep nvidia-smi compute apps on the relevant GPUs.
        """
        out: list[tuple[str, int]] = []
        with self._lock:
            for inst in self._instances.values():
                if inst.state == InstanceState.DEAD:
                    continue
                proc = getattr(inst.manager, "process", None)
                pid = getattr(proc, "pid", None)
                if pid is None:
                    continue
                try:
                    out.append((inst.instance_id, int(pid)))
                except (TypeError, ValueError):
                    continue
        return out

    def pause_starts(self) -> None:
        """Block subsequent ``start_instance`` calls.

        Called by the EmergencyGuard at the very start of its restart
        sequence so the main loop can't spawn fresh CARLAs that would
        survive the kill phase as orphans (we observed VRAM ascending
        4.4 → 8.7 GB during a drain wait because of this race).  Not
        un-paused: the master is dying anyway.
        """
        with self._lock:
            self._starts_paused = True

    def find_available_instance(
        self,
        *,
        town: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ) -> Optional[PoolInstance]:
        """Return an IDLE/CLEAN instance matching the given filters, or None.

        Priority: CLEAN same-town > CLEAN any town > IDLE (no town yet).
        If ``gpu_id`` is specified, only instances on that GPU are considered.
        """
        with self._lock:
            candidates = [
                i for i in self._instances.values()
                if i.state in (InstanceState.CLEAN, InstanceState.IDLE)
                and (gpu_id is None or i.gpu_id == gpu_id)
            ]
            if not candidates:
                return None
            # Priority sort: same-town CLEAN > any-town CLEAN > IDLE.
            def rank(i: PoolInstance) -> tuple[int, float]:
                if town is not None and i.current_town == town and i.state == InstanceState.CLEAN:
                    return (0, i.created_at)
                if i.state == InstanceState.CLEAN:
                    return (1, i.created_at)
                return (2, i.created_at)
            candidates.sort(key=rank)
            return candidates[0]

    # ------------------------------------------------------------------
    # Public: lease API
    # ------------------------------------------------------------------

    def acquire_lease(
        self,
        instance_id: str,
        town: str,
    ) -> Optional[CarlaLease]:
        """Grant an exclusive lease on a specific instance for a scenario.

        Caller must have previously obtained ``instance_id`` from
        :meth:`find_available_instance` or :meth:`start_instance`.

        Before handing the lease over, we run a fast RPC liveness probe
        against the instance.  A warm CARLA can die silently between
        scenarios (UE4 segfault, driver reset, OOM-kill by the OS, or a
        prior tenant leaving it in a wedged state).  Detecting that here
        -- BEFORE the grandchild spawns and burns a retry -- lets the
        scheduler requeue the job onto a fresh CARLA without counting it
        as a failed attempt.  On liveness failure we mark the instance
        for recycle and return None so the scheduler picks another.

        Town tracking: the pool records the town the lease is declared for,
        but does NOT itself issue ``world.load_world(town)``.  The grandchild
        evaluator already loads the target town as part of its own agent
        setup flow, so a pool-level load_world would be redundant and would
        also require importing the heavyweight CARLA Python API into the
        parent process.  If the grandchild fails to connect / load, it
        exits non-zero and the scheduler recycles the instance.
        """
        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                return None
            if inst.state not in (InstanceState.IDLE, InstanceState.CLEAN):
                log.warning(
                    "carla_pool: cannot lease instance=%s in state=%s",
                    instance_id,
                    inst.state,
                )
                return None
            # Defensive: process still alive?  A dead Popen means the
            # UE4 server segfaulted between leases; treat as crashed.
            try:
                proc = getattr(inst.manager, "process", None)
                if proc is not None and proc.poll() is not None:
                    log.warning(
                        "carla_pool: instance=%s process exited with "
                        "returncode=%s; recycling before lease",
                        instance_id,
                        proc.poll(),
                    )
                    inst.state = InstanceState.RECYCLING
                    inst._lease_id = None
            except Exception:
                pass
            if inst.state == InstanceState.RECYCLING:
                # Release lock before heavy stop_instance()
                pass

        # Liveness probe outside the pool lock so a slow RPC does not
        # block other scheduler decisions.
        if self._instances.get(instance_id) is not None:
            inst = self._instances[instance_id]
            if inst.state in (InstanceState.IDLE, InstanceState.CLEAN):
                rce = _rce()
                probe_timeout = max(3.0, float(self._rpc_ready_timeout_s) / 10.0)
                try:
                    if not rce.wait_for_carla_rpc_ready(
                        inst.host,
                        inst.world_port,
                        timeout=probe_timeout,
                        quiet=True,
                    ):
                        log.warning(
                            "carla_pool: lease-time RPC probe failed for "
                            "instance=%s host=%s port=%d -- recycling",
                            instance_id,
                            inst.host,
                            inst.world_port,
                        )
                        with self._lock:
                            inst2 = self._instances.get(instance_id)
                            if inst2 is not None and inst2.state in (
                                InstanceState.IDLE,
                                InstanceState.CLEAN,
                            ):
                                inst2.state = InstanceState.RECYCLING
                                inst2._lease_id = None
                        # Synchronously recycle so the scheduler doesn't
                        # see this zombie in find_available_instance().
                        try:
                            self.stop_instance(instance_id)
                        except Exception as exc:
                            log.warning(
                                "carla_pool: stop_instance after failed probe "
                                "raised for %s: %s",
                                instance_id,
                                exc,
                            )
                        self._emit(
                            "carla_lease_probe_failed",
                            instance_id=instance_id,
                            gpu_id=inst.gpu_id,
                            world_port=inst.world_port,
                        )
                        return None
                except Exception as exc:
                    log.warning(
                        "carla_pool: wait_for_carla_rpc_ready raised on "
                        "instance=%s: %s",
                        instance_id,
                        exc,
                    )

        with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                return None
            if inst.state not in (InstanceState.IDLE, InstanceState.CLEAN):
                return None
            inst.current_town = town

            lease = CarlaLease(
                lease_id=mint_lease_id(),
                instance_id=instance_id,
                gpu_id=inst.gpu_id,
                host=inst.host,
                world_port=inst.world_port,
                tm_port=inst.tm_port,
                current_town=inst.current_town,
                generation=inst.generation,
                acquired_at=time.monotonic(),
            )
            inst.state = InstanceState.LEASED
            inst._lease_id = lease.lease_id
            self._leases[lease.lease_id] = lease
        log.info(
            "carla_pool: lease acquired id=%s instance=%s town=%s",
            lease.lease_id,
            instance_id,
            town,
        )
        self._emit(
            "carla_lease_acquired",
            lease_id=lease.lease_id,
            instance_id=instance_id,
            gpu_id=inst.gpu_id,
            world_port=inst.world_port,
            tm_port=inst.tm_port,
            town=town,
            generation=inst.generation,
        )
        return lease

    def release_lease(
        self,
        lease: CarlaLease,
        status: LeaseStatus,
    ) -> None:
        """Return a lease.

        * ``clean``  : instance transitions LEASED -> CLEAN, ready for next
          acquire.  No teardown; town stays loaded.
        * ``dirty``  : Phase 3/4 will run a defensive world reset here; for
          now treated as ``clean``.
        * ``crashed``: instance transitions to RECYCLING; pool kills the
          process and waits for port close.  Removed from the pool.
        """
        with self._lock:
            inst = self._instances.get(lease.instance_id)
            if inst is None:
                log.warning(
                    "carla_pool: release of lease=%s on unknown instance=%s",
                    lease.lease_id,
                    lease.instance_id,
                )
                return
            if inst._lease_id != lease.lease_id:
                log.warning(
                    "carla_pool: double-release or stale lease=%s (instance holds %s)",
                    lease.lease_id,
                    inst._lease_id,
                )
                return
            inst.last_release_status = status
            inst.scenarios_served += 1
            inst._lease_id = None
            self._leases.pop(lease.lease_id, None)

            force_recycle = self._no_reuse or (status == "crashed")
            if force_recycle:
                inst.state = InstanceState.RECYCLING
            elif status == "dirty":
                # Phase 3: transition to RESETTING and run defensive cleanup
                # synchronously here.  For Phase 1, treat as clean.
                inst.state = InstanceState.CLEAN
            else:
                inst.state = InstanceState.CLEAN
            # Record the release timestamp so the idle-recycle reaper
            # knows how long this instance has been warm-but-unused.
            inst.last_released_at = time.monotonic()

        log.info(
            "carla_pool: lease released id=%s instance=%s status=%s -> state=%s",
            lease.lease_id,
            lease.instance_id,
            status,
            inst.state.value,
        )
        self._emit(
            "carla_lease_released",
            lease_id=lease.lease_id,
            instance_id=lease.instance_id,
            gpu_id=inst.gpu_id,
            world_port=inst.world_port,
            tm_port=inst.tm_port,
            town=inst.current_town,
            generation=inst.generation,
            status=status,
            recycled=force_recycle,
            next_state=inst.state.value,
        )
        if force_recycle:
            # Stop synchronously so the scheduler can redispatch on this
            # GPU once VRAM frees.  In no-reuse mode we recycle on EVERY
            # release; otherwise only on crashes.
            self.stop_instance(inst.instance_id)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Terminate all instances and release all resources.

        Also best-effort hunts down orphan CARLA server processes that
        were started by *this* pool but whose manager handle has already
        been lost (happens when a kernel OOM-killer or SIGKILL reaps the
        bash wrapper before this shutdown fires).  Uses the
        ``--world-port=N`` command-line fingerprint to match, so we only
        touch PIDs whose port band we minted.
        """
        # Stop the idle-recycle reaper first so it doesn't race with us
        # on the instances we're about to stop synchronously.
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            try:
                self._reaper_thread.join(timeout=self._idle_recycle_poll_seconds + 5.0)
            except Exception:
                pass
        with self._lock:
            ids = list(self._instances.keys())
            known_ports = {inst.world_port for inst in self._instances.values()}
        for instance_id in ids:
            try:
                # If a lease is still out, detach it first.
                with self._lock:
                    inst = self._instances.get(instance_id)
                    if inst is not None and inst.state == InstanceState.LEASED:
                        inst.state = InstanceState.RECYCLING
                        inst._lease_id = None
                self.stop_instance(instance_id)
            except Exception as exc:
                log.warning("carla_pool: shutdown error for %s: %s", instance_id, exc)

        # Second pass: kill any CARLA server still bound to one of the
        # ports we owned.  This closes the window where ``stop_instance``
        # thought the process was gone but the CarlaUE4-Linux-Shipping
        # child survived (happens when the bash wrapper exits but the
        # Unreal binary double-forked).
        if not known_ports:
            return
        try:
            import re
            import subprocess
            proc = subprocess.run(
                ["ps", "-eo", "pid,command"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
            for line in proc.stdout.splitlines():
                line = line.strip()
                if "CarlaUE4" not in line and "leaderboard_evaluator" not in line:
                    continue
                # Parse --world-port=N or --port N
                m = re.search(r"--world-port=(\d+)|--port\s+(\d+)", line)
                if not m:
                    continue
                port = int(m.group(1) or m.group(2))
                if port not in known_ports:
                    continue
                try:
                    pid = int(line.split(None, 1)[0])
                except Exception:
                    continue
                log.warning(
                    "carla_pool: shutdown sweep killing orphan pid=%s port=%s "
                    "cmd=%s",
                    pid, port, line[:120],
                )
                try:
                    import os, signal as _sig
                    os.killpg(os.getpgid(pid), _sig.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    log.warning(
                        "carla_pool: no permission to signal pid=%s; skipping",
                        pid,
                    )
                except Exception as exc:
                    log.warning(
                        "carla_pool: signal pid=%s raised: %s",
                        pid, exc,
                    )
        except Exception:
            log.exception("carla_pool: shutdown orphan-sweep failed")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_free_port(self, gpu_id: int) -> Optional[tuple[int, int]]:
        """Walk the GPU's candidate band, return the first pair whose ports
        are confirmed free.  Also excludes ports currently held by pool
        instances to prevent same-process collisions.
        """
        rce = _rce()
        occupied: set[int] = set()
        with self._lock:
            for other in self._instances.values():
                # Any non-DEAD instance's ports are off-limits.
                if other.state != InstanceState.DEAD:
                    occupied.add(other.world_port)
                    occupied.add(other.tm_port)

        for world_port, tm_port in self._port_band_fn(gpu_id):
            if world_port in occupied or tm_port in occupied:
                continue
            ports_to_check = rce.carla_service_ports(world_port, self._tm_port_offset)
            if rce.are_ports_available(self._host, ports_to_check):
                return (world_port, tm_port)
        return None


def default_port_band(
    gpu_id: int,
    *,
    world_base: int = 2000,
    stride: int = 4,
    slots: int = 64,
    per_gpu_span: int = 512,
) -> list[tuple[int, int]]:
    """Generate the default candidate port band for a GPU.

    Layout::

        gpu 0 -> world ports 2000, 2004, 2008, ..., 2252   (64 slots)
        gpu 1 -> world ports 2512, 2516, 2520, ..., 2764   (64 slots)
        gpu 2 -> world ports 3024, ...
        ...

    ``tm_port`` is always ``world_port + 2`` by convention.  The pool
    verifies each candidate is free before binding; a candidate whose ports
    are occupied is silently skipped.

    ``slots`` is the candidate count per GPU -- NOT a concurrency cap.  The
    scheduler (which owns VRAM accounting) decides how many instances are
    actually started on each GPU; as long as at least one candidate in the
    band is free, a new instance can be created.  The default (64 slots,
    512-port span per GPU) is intentionally generous: it makes "no free
    port in band" unreachable under normal concurrency and still leaves
    TIME_WAIT headroom after instances are torn down, without colliding
    with the next GPU's band.  The linear walk in ``_find_free_port`` is
    O(slots) which is fine at this size.
    """
    start = world_base + gpu_id * per_gpu_span
    assert stride * slots <= per_gpu_span, "slots * stride exceeds per-GPU span"
    return [(start + i * stride, start + i * stride + 2) for i in range(slots)]


def mint_lease_id() -> str:
    return f"lease-{uuid.uuid4().hex[:10]}"


def mint_instance_id(gpu_id: int, seq: int) -> str:
    return f"carla-gpu{gpu_id}-{seq:04d}"
