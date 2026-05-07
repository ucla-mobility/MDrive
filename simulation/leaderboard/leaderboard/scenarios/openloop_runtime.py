"""
Open-loop replay runtime for the leaderboard scenario manager.

Currently DORMANT in the standard workflow. The scenario manager only
constructs an ``OpenLoopRuntime`` when BOTH ``CUSTOM_EGO_LOG_REPLAY=1``
AND ``CUSTOM_OPENLOOP_TELEPORT=1`` are set. The default ``--openloop``
mode (frame-0 prediction dump + closed-loop drive) sets only the first
env var and never activates this module. Set ``CUSTOM_OPENLOOP_TELEPORT=1``
to opt back into the legacy full-trajectory teleport behaviour described
below. The integration is untested in current experiments and may bit-rot.

Purpose
-------
When the launcher passes ``--openloop`` (which sets ``CUSTOM_EGO_LOG_REPLAY=1``)
AND ``CUSTOM_OPENLOOP_TELEPORT=1`` is set, the ego must be driven *exclusively*
from the recorded ``_REPLAY.xml`` trajectory. The agent's ``run_step`` still
fires every tick (so perception + per-tick predictions are produced), but its
returned ``(steer, throttle, brake)`` MUST NOT influence the ego's pose. That
guarantees the trajectory is byte-identical across detector swaps, which is
the whole point of an open-loop comparison.

Historical leak this module fixes
---------------------------------
Prior to this module:
  * The XML discovery in ``leaderboard_evaluator_parameter.py`` mapped
    ``..._0_REPLAY.xml`` to ``ego_id_str = "REPLAY"`` (filename split bug),
    so the dense replay never reached ``config.trajectory_times``.
  * ``RouteScenario._build_ego_replay_plans`` therefore set
    ``_ego_replay_times[i] = None`` for every ego.
  * ``RouteScenario._create_behavior`` skipped attaching the
    ``LogReplayEgo`` follower (its ``if transforms and times`` guard failed).
  * Result: ego ran fully closed-loop. Across detector swaps, trajectories
    diverged within 4 ticks (0.2 s).

This module is the belt-and-braces second layer: even if the behavior tree
ever stops calling ``set_transform`` for the ego, the scenario manager
will force-teleport every tick from the parsed replay and bail with a clean
status the moment the replay XML is exhausted.

Public API
----------
``OpenLoopRuntime`` is the single object the scenario manager owns.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import carla


# Env-var-tunable thresholds. Defaults chosen so a healthy run never warns.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


# Pose drift tolerances. Per-tick CARLA physics integration with velocity
# reset to 0 leaves at most ~3 cm of slop from the queued control's impulse
# at full throttle on one 0.05 s frame; 5 cm is the warn floor and 1 m the
# raise floor (3 sigma above any plausible drift = real bug).
OPENLOOP_POSE_WARN_M = _env_float("OPENLOOP_POSE_WARN_M", 0.05)
OPENLOOP_POSE_ABORT_M = _env_float("OPENLOOP_POSE_ABORT_M", 1.0)

# Spot-check at most this many NPC actors per tick. Full-fleet checks at 88
# actors × 20 Hz are cheap but unnecessary noise; 5 random samples confirm
# the LogReplayActor wiring without flooding logs.
OPENLOOP_NPC_SPOT_CHECK = _env_int("OPENLOOP_NPC_SPOT_CHECK", 5)

# Run an extra tick after the replay end so the agent emits one final
# prediction frame at the terminal pose. Set to 0 to terminate immediately.
OPENLOOP_TAIL_TICKS = _env_int("OPENLOOP_TAIL_TICKS", 0)


class OpenLoopIntegrityError(RuntimeError):
    """Raised when the actual ego pose diverges past OPENLOOP_POSE_ABORT_M
    from the expected REPLAY pose. Indicates an open-loop leak (planner
    controls reaching the actuator) or a CARLA pose-application failure."""


@dataclass
class _Waypoint:
    t: float            # sim seconds since scenario start
    x: float
    y: float
    z: float
    yaw_deg: float      # CARLA yaw, degrees
    pitch_deg: float
    roll_deg: float


def _shortest_arc_deg(a: float, b: float, frac: float) -> float:
    """Interpolate b toward a along the short arc (degrees)."""
    delta = (b - a + 180.0) % 360.0 - 180.0
    return a + frac * delta


class _ReplayPlan:
    """Per-ego dense trajectory loaded from an XML's parsed ``trajectory`` +
    ``trajectory_times`` lists. Provides ``transform_at(t)`` via linear
    interpolation in xyz and shortest-arc interpolation in yaw."""

    def __init__(self, waypoints: List[_Waypoint]):
        if not waypoints:
            raise ValueError("Empty replay plan")
        # Sort defensively. The XML is normally already in time order.
        waypoints.sort(key=lambda w: w.t)
        self._wp = waypoints
        # Pre-extract times for fast bisect.
        self._times = [w.t for w in waypoints]
        self.t_start = waypoints[0].t
        self.t_end = waypoints[-1].t

    @classmethod
    def from_config(
        cls,
        traj: Optional[List[carla.Location]],
        times: Optional[List[Optional[float]]],
        yaws: Optional[List[Optional[float]]] = None,
        pitches: Optional[List[Optional[float]]] = None,
        rolls: Optional[List[Optional[float]]] = None,
    ) -> Optional["_ReplayPlan"]:
        if not traj or not times:
            return None
        if len(traj) != len(times):
            return None
        # Reject replays missing any timestamp — without complete timing we
        # can't pin the actor to a specific sim time.
        if not all(t is not None for t in times):
            return None
        wps: List[_Waypoint] = []
        for i, loc in enumerate(traj):
            t = float(times[i])
            yaw_val = yaws[i] if (yaws is not None and i < len(yaws)) else None
            pitch_val = pitches[i] if (pitches is not None and i < len(pitches)) else None
            roll_val = rolls[i] if (rolls is not None and i < len(rolls)) else None
            if yaw_val is None:
                # Derive yaw from segment direction if missing.
                if i + 1 < len(traj):
                    nxt = traj[i + 1]
                    yaw_val = math.degrees(math.atan2(nxt.y - loc.y, nxt.x - loc.x))
                elif i > 0:
                    prv = traj[i - 1]
                    yaw_val = math.degrees(math.atan2(loc.y - prv.y, loc.x - prv.x))
                else:
                    yaw_val = 0.0
            wps.append(_Waypoint(
                t=t,
                x=float(loc.x), y=float(loc.y), z=float(loc.z),
                yaw_deg=float(yaw_val),
                pitch_deg=float(pitch_val) if pitch_val is not None else 0.0,
                roll_deg=float(roll_val) if roll_val is not None else 0.0,
            ))
        if len(wps) < 1:
            return None
        return cls(wps)

    def transform_at(self, t: float) -> carla.Transform:
        """Linear interpolation in xyz, shortest-arc in yaw/pitch/roll.
        Outside the recorded interval the endpoint pose is returned (caller
        should also check `exhausted_at(t)` to decide when to stop)."""
        wp = self._wp
        n = len(wp)
        if n == 1:
            w = wp[0]
            return carla.Transform(
                carla.Location(x=w.x, y=w.y, z=w.z),
                carla.Rotation(pitch=w.pitch_deg, yaw=w.yaw_deg, roll=w.roll_deg),
            )
        if t <= self.t_start:
            w = wp[0]
            return carla.Transform(
                carla.Location(x=w.x, y=w.y, z=w.z),
                carla.Rotation(pitch=w.pitch_deg, yaw=w.yaw_deg, roll=w.roll_deg),
            )
        if t >= self.t_end:
            w = wp[-1]
            return carla.Transform(
                carla.Location(x=w.x, y=w.y, z=w.z),
                carla.Rotation(pitch=w.pitch_deg, yaw=w.yaw_deg, roll=w.roll_deg),
            )
        # bisect for the segment containing t
        # Manual binary search (no `bisect` import for forward portability with
        # the parsed-list layout above).
        lo, hi = 0, n - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._times[mid] <= t:
                lo = mid
            else:
                hi = mid
        a, b = wp[lo], wp[hi]
        span = b.t - a.t
        frac = 0.0 if span <= 1e-9 else (t - a.t) / span
        x = a.x + frac * (b.x - a.x)
        y = a.y + frac * (b.y - a.y)
        z = a.z + frac * (b.z - a.z)
        yaw = _shortest_arc_deg(a.yaw_deg, b.yaw_deg, frac)
        pitch = _shortest_arc_deg(a.pitch_deg, b.pitch_deg, frac)
        roll = _shortest_arc_deg(a.roll_deg, b.roll_deg, frac)
        return carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
        )

    def velocity_at(self, t: float) -> Tuple[float, float, float]:
        """World-frame (vx, vy, vz) m/s at sim time ``t``.

        Computed from the segment containing ``t`` as a finite difference
        ``(b - a) / (t_b - t_a)``. Outside the recorded interval the segment
        velocity at the closest endpoint is returned.
        """
        wp = self._wp
        n = len(wp)
        if n < 2:
            return (0.0, 0.0, 0.0)
        if t <= self.t_start:
            a, b = wp[0], wp[1]
        elif t >= self.t_end:
            a, b = wp[-2], wp[-1]
        else:
            lo, hi = 0, n - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self._times[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            a, b = wp[lo], wp[hi]
        span = b.t - a.t
        if span <= 1e-9:
            return (0.0, 0.0, 0.0)
        return (
            (b.x - a.x) / span,
            (b.y - a.y) / span,
            (b.z - a.z) / span,
        )

    def exhausted_at(self, t: float) -> bool:
        return t > self.t_end


class _NpcSpotCheck:
    """Lightweight sample of NPC replay plans. We don't *drive* the NPCs —
    the existing LogReplayActor followers do — but we sample a few of them
    each tick to verify the LogReplayActor wiring is also doing its job."""

    def __init__(self, plans: List[Tuple[str, Any, _ReplayPlan]]):
        # Each entry: (npc_name, actor_handle, replay_plan)
        self._plans = plans
        self._rng = random.Random(0xC0DE)
        self._max_samples = OPENLOOP_NPC_SPOT_CHECK

    def sample(self) -> List[Tuple[str, Any, _ReplayPlan]]:
        if not self._plans:
            return []
        n = min(self._max_samples, len(self._plans))
        return self._rng.sample(self._plans, n)


class OpenLoopRuntime:
    """Single object owned by the ScenarioManager when --openloop is on.

    Responsibilities:
      1. Keep the per-ego ``_ReplayPlan``s.
      2. ``ensure_pose(...)`` — at the end of every tick, force-teleport
         each ego to ``transform_at(sim_t)`` and zero velocity / control.
      3. ``check_pose(...)`` — at the start of each tick, before the agent
         runs, verify the actor is where the replay says it should be.
         Warn at >= OPENLOOP_POSE_WARN_M, raise OpenLoopIntegrityError at
         >= OPENLOOP_POSE_ABORT_M.
      4. ``all_exhausted(sim_t)`` — True once every ego has passed its
         replay end so the manager can terminate cleanly.
    """

    def __init__(self, ego_plans: List[Optional[_ReplayPlan]]):
        self._ego_plans: List[Optional[_ReplayPlan]] = list(ego_plans)
        self._npcs: Optional[_NpcSpotCheck] = None
        self._enable_correctness = _env_flag("OPENLOOP_CORRECTNESS_CHECK", True)
        self._abort_on_drift = _env_flag("OPENLOOP_ABORT_ON_DRIFT", True)
        # Bookkeeping for clean termination.
        self._tail_remaining = OPENLOOP_TAIL_TICKS
        # Diagnostics counters.
        self._n_warnings = 0
        self._n_force_teleports = 0
        # Per-ego previous expected_pose, for diag messages.
        self._last_expected: Dict[int, carla.Transform] = {}

    @classmethod
    def from_scenario_config(cls, scenario_config) -> Optional["OpenLoopRuntime"]:
        """Build from a config carrying ``multi_replay_traj``/``multi_replay_times``
        (populated by ``leaderboard_evaluator_parameter.py``). Returns None
        if no ego has a usable replay — in that case the manager skips
        openloop enforcement."""
        traj = getattr(scenario_config, "multi_replay_traj", None)
        times = getattr(scenario_config, "multi_replay_times", None)
        yaws = getattr(scenario_config, "multi_replay_yaws", None)
        pitches = getattr(scenario_config, "multi_replay_pitches", None)
        rolls = getattr(scenario_config, "multi_replay_rolls", None)
        if not traj or not times:
            return None
        ego_plans: List[Optional[_ReplayPlan]] = []
        for i in range(len(traj)):
            plan = _ReplayPlan.from_config(
                traj[i],
                times[i],
                yaws[i] if (yaws is not None and i < len(yaws)) else None,
                pitches[i] if (pitches is not None and i < len(pitches)) else None,
                rolls[i] if (rolls is not None and i < len(rolls)) else None,
            )
            ego_plans.append(plan)
        if all(p is None for p in ego_plans):
            return None
        return cls(ego_plans)

    # -------------------------------------------------------------- query
    def has_plan_for(self, ego_idx: int) -> bool:
        return 0 <= ego_idx < len(self._ego_plans) and self._ego_plans[ego_idx] is not None

    def expected_pose(self, ego_idx: int, sim_t: float) -> Optional[carla.Transform]:
        if not self.has_plan_for(ego_idx):
            return None
        return self._ego_plans[ego_idx].transform_at(sim_t)

    def expected_velocity(self, ego_idx: int, sim_t: float) -> Optional[Tuple[float, float, float]]:
        if not self.has_plan_for(ego_idx):
            return None
        return self._ego_plans[ego_idx].velocity_at(sim_t)

    def all_exhausted(self, sim_t: float) -> bool:
        """True iff every ego with a replay plan has passed t_end."""
        any_active = False
        for plan in self._ego_plans:
            if plan is None:
                continue
            any_active = True
            if not plan.exhausted_at(sim_t):
                return False
        return any_active  # only signal exhaustion if there *was* a plan

    def consume_tail_tick(self) -> bool:
        """Returns True while the optional tail-tick budget is non-zero, so
        the caller can grant one more agent tick after exhaustion."""
        if self._tail_remaining <= 0:
            return False
        self._tail_remaining -= 1
        return True

    # --------------------------------------------------- per-tick actions
    def ensure_pose(self, ego_actors: List[Any], sim_t_next: float) -> None:
        """End-of-tick force-teleport. Called AFTER apply_control + scenario
        tree tick so the actor is pinned to ``transform_at(sim_t_next)`` in
        time for the next world.tick()."""
        # One-time diagnostic: log the actor↔plan mapping on first call so we
        # can verify ego_vehicles[idx] actually matches _ego_plans[idx]. Bug
        # symptom we hit: ego_vehicles ordering didn't match plan ordering →
        # ensure_pose teleported actor 0 to plan 0's target (works), but the
        # actor at index 1 stayed at its spawn pose because we either didn't
        # touch it (plan was None or actor was None at that index) or
        # touched the wrong thing.
        if not getattr(self, "_logged_actor_plan_map", False):
            self._logged_actor_plan_map = True
            for idx in range(max(len(ego_actors), len(self._ego_plans))):
                actor = ego_actors[idx] if idx < len(ego_actors) else None
                plan = self._ego_plans[idx] if idx < len(self._ego_plans) else None
                actor_id = getattr(actor, "id", None)
                plan_t0 = (plan._wp[0].x, plan._wp[0].y) if plan else None
                actor_loc = None
                try:
                    if actor is not None:
                        loc = actor.get_transform().location
                        actor_loc = (loc.x, loc.y)
                except Exception:
                    pass
                print(f"[OPENLOOP] map idx={idx} actor_id={actor_id} "
                      f"actor_loc={actor_loc} plan_t0={plan_t0}")
        for idx, actor in enumerate(ego_actors):
            if actor is None or not getattr(actor, "is_alive", True):
                continue
            target = self.expected_pose(idx, sim_t_next)
            if target is None:
                continue
            try:
                actor.set_transform(target)
                # Set CARLA velocity to the trajectory-derived GT velocity so
                # carla_speedometer reports the recorded expert speed (instead
                # of zero) to the agent. This is what feeds ``ego.speed`` in
                # the planner — without it the planner thinks the car is
                # parked and produces a stationary prediction.
                vel = self.expected_velocity(idx, sim_t_next)
                if vel is None:
                    vel = (0.0, 0.0, 0.0)
                actor.set_target_velocity(carla.Vector3D(vel[0], vel[1], vel[2]))
                actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                self._n_force_teleports += 1
            except Exception as exc:
                # Don't raise here — the correctness check on the next tick
                # will catch and surface the divergence.
                print(f"[OPENLOOP] ego{idx}: set_transform failed: {exc}")

    def check_pose(self, ego_actors: List[Any], sim_t: float) -> None:
        """Top-of-tick correctness verification. Compares the actor's actual
        pose against the replay-expected pose at ``sim_t`` (before the agent
        runs). Warns >=OPENLOOP_POSE_WARN_M; raises >=OPENLOOP_POSE_ABORT_M."""
        if not self._enable_correctness:
            return
        for idx, actor in enumerate(ego_actors):
            if actor is None or not getattr(actor, "is_alive", True):
                continue
            expected = self.expected_pose(idx, sim_t)
            if expected is None:
                continue
            try:
                actual = actor.get_transform()
            except Exception:
                continue
            err_xy = math.hypot(
                actual.location.x - expected.location.x,
                actual.location.y - expected.location.y,
            )
            err_z = abs(actual.location.z - expected.location.z)
            if err_xy >= OPENLOOP_POSE_ABORT_M:
                msg = (
                    f"[OPENLOOP] ego{idx} t={sim_t:.3f}s pose-divergence "
                    f"err_xy={err_xy:.3f}m err_z={err_z:.3f}m "
                    f"actual=({actual.location.x:.2f},{actual.location.y:.2f},{actual.location.z:.2f}) "
                    f"expected=({expected.location.x:.2f},{expected.location.y:.2f},{expected.location.z:.2f}) "
                    f"— OPEN-LOOP LEAK"
                )
                if self._abort_on_drift:
                    raise OpenLoopIntegrityError(msg)
                print(msg)
            elif err_xy >= OPENLOOP_POSE_WARN_M:
                self._n_warnings += 1
                # Throttle: print first 5, then every 100th.
                if self._n_warnings <= 5 or (self._n_warnings % 100 == 0):
                    print(
                        f"[OPENLOOP-WARN] ego{idx} t={sim_t:.3f}s err_xy={err_xy:.3f}m "
                        f"err_z={err_z:.3f}m (count={self._n_warnings})"
                    )
            self._last_expected[idx] = expected

        # NPC spot-check
        if self._npcs is not None:
            for name, actor, plan in self._npcs.sample():
                if actor is None or not getattr(actor, "is_alive", True):
                    continue
                exp = plan.transform_at(sim_t)
                try:
                    act = actor.get_transform()
                except Exception:
                    continue
                err_xy = math.hypot(
                    act.location.x - exp.location.x,
                    act.location.y - exp.location.y,
                )
                if err_xy >= OPENLOOP_POSE_WARN_M:
                    print(
                        f"[OPENLOOP-NPC-WARN] {name} t={sim_t:.3f}s "
                        f"err_xy={err_xy:.3f}m"
                    )

    # --------------------------------------------------------- diagnostics
    def stats(self) -> Dict[str, Any]:
        return {
            "n_warnings": self._n_warnings,
            "n_force_teleports": self._n_force_teleports,
            "n_egos_with_plan": sum(1 for p in self._ego_plans if p is not None),
        }
