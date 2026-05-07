"""
BasicAgentPlanner — wraps CARLA's built-in BasicAgent so its hazard check
consumes our Detection list (perception output OR ground-truth oracle) instead
of querying ``world.get_actors()``.

This is the experimental scaffold for the perception-as-only-independent-variable
study. CARLA's lateral PID, longitudinal PID, route following, and traffic-light
handling are all reused unchanged — they are battle-tested in the Leaderboard.
The ONE thing that varies between matrix cells is the Detection list passed in:

    perception output  → real run     (bottleneck = detector quality)
    world.get_actors() → GT oracle    (bottleneck = planner only — the control row)

Compared to RulePlanner this trades the home-grown 50-line steering+brake logic
for CARLA's PID controllers. That isolates the comparison: any closed-loop
failure in a "real" detector run that does NOT also fail under GT is attributable
to perception, not the planner.

Usage
-----
    planner = BasicAgentPlanner(actor=hero_vehicle, target_speed_kmh=25)
    planner.set_global_plan(world_plan)            # [(carla.Transform, RoadOption), ...]

    # Per step:
    planner.set_detections(detections_ego_frame)   # List[Detection]
    control = planner.run_step()                   # carla.VehicleControl
    traj    = planner.predict_trajectory(3.0, 0.1) # (T,2) world-frame xy
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import carla
import numpy as np

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

from team_code.rule_planner import Detection


# Half-width of a CARLA driving lane in meters; matches RulePlanner's corridor.
DEFAULT_LANE_HALF_WIDTH = 1.6


class BasicAgentPlanner:
    """CARLA BasicAgent with its hazard check rewired to consume a Detection list.

    Detections are in EGO frame (CARLA LH: +x forward, +y right). That matches
    HEAL's perception output natively, and the GT-oracle path projects world
    actors into the same frame — so the planner's view is convention-stable
    across matrix cells.
    """

    def __init__(
        self,
        actor: carla.Vehicle,
        target_speed_kmh: float = 25.0,
        corridor_half_width: float = DEFAULT_LANE_HALF_WIDTH,
        brake_persistence_ticks: int = 3,
        lateral_clearance_m: float = 0.9,
        stopped_speed_m_s: float = 0.3,
        stopped_min_forward_m: float = 4.0,
        opt_dict: Optional[dict] = None,
        colmdriver_aligned: bool = False,
    ):
        if actor is None:
            raise ValueError("BasicAgentPlanner: actor is None — defer construction until hero is spawned")
        self._actor = actor
        self._corridor_half_width = float(corridor_half_width)
        self._detections: List[Detection] = []
        # Detection-persistence filter (same idea as RulePlanner): an in-corridor
        # detection must persist for N consecutive ticks before the override
        # asserts hazard_detected=True. Filters single-frame perception FPs
        # which previously locked the ego in dense walker scenes.
        self._brake_persistence_ticks = max(1, int(brake_persistence_ticks))
        self._brake_streak = 0
        # Realistic vehicle half-width for the lateral "in our lane?" check —
        # replaces the legacy det.radius (1.5m) which inflated the corridor
        # to 6.2m total and treated parked side-cars at 2-3m lateral as
        # in-lane blockers (observed: cobevt 11_0 ego1 stuck 800+ ticks).
        self._lateral_clearance_m = float(lateral_clearance_m)
        # Stationary-ego bypass — when ego.speed is below stopped_speed_m_s
        # AND obstacle is at least stopped_min_forward_m forward, skip the
        # brake decision. Without this, the BasicAgent gets locked next to
        # any stationary obstacle: corridor stays True forever, brake_streak
        # builds, emergency-stop returned every tick, ego never moves.
        self._stopped_speed_m_s = float(stopped_speed_m_s)
        self._stopped_min_forward_m = float(stopped_min_forward_m)
        # NOTE: A "stuck-too-long" force-release was tried in iter2 (5s of
        # in_corridor + ego stopped → one-tick bypass). It HURT 3 of 4
        # detectors (cob -0.9, dis -4.2, fco -1.1) because the bypass let
        # ego creep into real obstacles. Removed; iter1 (no stuck recovery)
        # is the final baseline.

        opt = dict(opt_dict or {})
        # Slightly larger than BasicAgent's default 5 m so we react earlier at
        # cruise speed (the threshold is then padded by current vehicle_speed).
        opt.setdefault("base_vehicle_threshold", 6.0)
        if colmdriver_aligned:
            # HITL only — match CoLMDriver's V2X_Controller config
            # (colmdriver_config.yaml): max_throttle 0.75 + longitudinal PID
            # K_P=5.0/K_I=1.0/K_D=0.1, so HITL's cruise/acceleration profile
            # is byte-for-byte identical to CoLM on the longitudinal axis.
            # Other consumers (perception_swap) keep BasicAgent's defaults.
            opt.setdefault("max_throttle", 0.75)
            opt.setdefault("longitudinal_control_dict",
                           {"K_P": 5.0, "K_I": 1.0, "K_D": 0.1, "dt": 0.05})
        else:
            opt.setdefault("max_throttle", 0.6)
        opt.setdefault("max_brake", 0.5)
        opt.setdefault("max_steering", 0.8)
        # Lateral PID retune. CARLA's BasicAgent default is K_P=1.95 — at
        # spawn, with a 90° heading-to-route error and an empty error buffer,
        # the controller emits 1.95 × 1.57 ≈ 3.05 → clipped to 1.0 (full
        # right) → integrator winds up while the vehicle is yawing → the ego
        # over-rotates ~90° in 1.5 s and clips parked traffic.  K_P=0.5 keeps
        # the same large-error response below saturation (0.5 × 1.57 ≈ 0.79),
        # so the steer-rate limiter (±0.1/tick in VehiclePIDController) ramps
        # up gracefully instead of pinning at the clip.  K_I and K_D unchanged.
        opt.setdefault("lateral_control_dict",
                       {"K_P": 0.5, "K_I": 0.05, "K_D": 0.2, "dt": 0.05})

        self._agent = BasicAgent(actor, target_speed=target_speed_kmh, opt_dict=opt)

        # Replace the world-querying obstacle filter with our detection-driven one.
        # Bound-method assignment is safe: BasicAgent.run_step calls
        # `self._vehicle_obstacle_detected(...)` so attribute lookup finds ours.
        self._agent._vehicle_obstacle_detected = self._detection_obstacle_detected

    # ───── Wiring ─────────────────────────────────────────────────────

    def set_global_plan(self, world_plan):
        """world_plan: list of (carla.Transform, RoadOption) — leaderboard's format."""
        carla_map = self._actor.get_world().get_map()
        plan = []
        for transform, opt in world_plan:
            wp = carla_map.get_waypoint(transform.location)
            if wp is None:
                continue
            plan.append((wp, opt if opt is not None else RoadOption.LANEFOLLOW))
        if not plan:
            raise RuntimeError("BasicAgentPlanner: no waypoints survived map projection")
        self._agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)

    def set_detections(self, detections: List[Detection]):
        """Set the obstacle list for the next run_step. Detections are ego-frame."""
        self._detections = list(detections or [])

    # ───── Step ───────────────────────────────────────────────────────

    def _drop_behind_waypoints(self):
        """Remove waypoints from the front of the LocalPlanner queue that lie
        BEHIND the ego (negative dot-product with the ego forward vector).

        CARLA's stock ``LocalPlanner.run_step`` pops only by *distance* —
        ``if veh_location.distance(wp) < min_distance``. That works when the
        ego physically follows the route, but fails when the ego ends up some
        meters AHEAD of route[0] (e.g., scenario_runner's pre-tick physics
        steps move the ego ~5 m forward between spawn and the agent's first
        run_step in v2xpnp scenes).  In that case, route points 0..k can be
        further than ``min_distance`` *behind* the ego — so they stay at the
        head of the queue and the PIDLateralController tries to U-turn back
        to them.  The cross-z component degenerates to ~0 (target almost
        anti-parallel to forward), so the PID's sign rule defaults positive,
        producing sustained max-right steer — observed as a consistent CW
        rotation that clips parked traffic in the next lane.

        Fix: each tick, before BasicAgent.run_step runs, prune leading queue
        entries whose vector-from-ego has negative dot-product with the ego
        forward vector. The first AHEAD waypoint becomes the target.
        """
        try:
            queue = self._agent._local_planner._waypoints_queue
            if not queue:
                return
            tf = self._actor.get_transform()
            fx = float(np.cos(np.radians(tf.rotation.yaw)))
            fy = float(np.sin(np.radians(tf.rotation.yaw)))
            ex, ey = float(tf.location.x), float(tf.location.y)
            popped = 0
            while queue:
                wp = queue[0][0]
                wloc = wp.transform.location
                dx = float(wloc.x) - ex
                dy = float(wloc.y) - ey
                # forward·to-waypoint < 0 ⇒ waypoint is behind ego.
                # Use a small negative epsilon (-0.1 m) so we don't churn on
                # waypoints that are exactly perpendicular to forward.
                if (fx * dx + fy * dy) < -0.1:
                    queue.popleft()
                    popped += 1
                    if popped >= len(queue) + 1:
                        break
                else:
                    break
        except Exception:
            # Be defensive — never let queue cleanup break the run.
            pass

    def run_step(self) -> carla.VehicleControl:
        self._drop_behind_waypoints()
        # Goal-reached: BasicAgent's done() fires when waypoint queue empties
        # (ego has consumed the entire route). Without this, agent.run_step()
        # returns a default zero-throttle/zero-brake control and the ego
        # COASTS past the goal, accruing penalties / drifting off-route.
        # Issue an explicit full brake so RouteCompletionTest and CARLA
        # both see a stopped ego.
        if self._agent.done():
            ctrl = carla.VehicleControl()
            ctrl.steer = 0.0
            ctrl.throttle = 0.0
            ctrl.brake = 1.0
            ctrl.hand_brake = False
            return ctrl
        return self._agent.run_step()

    def done(self) -> bool:
        return self._agent.done()

    @property
    def target_speed_mps(self) -> float:
        return float(self._agent._local_planner._target_speed) / 3.6

    # ───── Trajectory rollout for open-loop ADE/FDE ───────────────────

    def predict_trajectory(self, horizon_s: float = 3.0, dt: float = 0.1) -> np.ndarray:
        """Walk the local planner's queued waypoints at target_speed.

        Returns (T, 2) world-frame xy. The rollout ramps from the ego's
        *current* speed up to ``target_speed_mps`` at a fixed acceleration
        (``a_max = 2.0 m/s²``) — the constant-target-speed rollout that
        existed previously produced an instantaneous 6 m/s jump on the very
        first tick (when the ego is still stationary), inflating ADE by
        ~10 m at tick 0 and making the per-tick BEV display useless during
        the warmup window. With ramping, ADE at tick 0 reflects the actual
        controller acceleration the ego is about to apply, not a phantom
        teleport.

        Override the ramp accel via env ``OPENLOOP_PRED_ACCEL_MPS2`` (set to
        a very large number to recover the legacy step-to-target behavior).
        """
        target = self.target_speed_mps
        try:
            v0 = float(self._actor.get_velocity().x ** 2
                       + self._actor.get_velocity().y ** 2) ** 0.5
        except Exception:
            v0 = target
        try:
            a_max = float(os.environ.get("OPENLOOP_PRED_ACCEL_MPS2", "2.0"))
        except Exception:
            a_max = 2.0
        queue = list(self._agent._local_planner._waypoints_queue)
        T = max(1, int(round(horizon_s / dt)))
        loc = self._actor.get_location()
        pos = np.array([loc.x, loc.y], dtype=np.float64)

        if not queue:
            return np.tile(pos, (T, 1))

        out = np.zeros((T, 2), dtype=np.float64)
        cursor = 0
        pop_dist = 1.5  # m, when to advance to next waypoint
        speed = float(v0)

        for t in range(T):
            # Ramp speed toward target at bounded accel.
            if speed < target:
                speed = min(target, speed + a_max * dt)
            elif speed > target:
                speed = max(target, speed - a_max * dt)
            while cursor < len(queue) - 1:
                wp = queue[cursor][0].transform.location
                if math.hypot(wp.x - pos[0], wp.y - pos[1]) < pop_dist:
                    cursor += 1
                else:
                    break
            wp = queue[cursor][0].transform.location
            vec = np.array([wp.x - pos[0], wp.y - pos[1]], dtype=np.float64)
            d = float(np.linalg.norm(vec))
            cur_dir = vec / d if d > 1e-6 else np.array([0.0, 0.0])
            pos = pos + cur_dir * speed * dt
            out[t] = pos
        return out

    # ───── The crucial swap: detection-driven hazard check ────────────

    def _detection_obstacle_detected(self, vehicle_list=None, max_distance=None):
        """Replacement for ``BasicAgent._vehicle_obstacle_detected``.

        BasicAgent calls this from ``run_step`` to decide whether to emergency-stop.
        The original implementation iterates ``world.get_actors().filter("*vehicle*")``
        and projects each onto the ego's lane via the CARLA map. Here we use the
        ego-frame Detection list directly:

          * x_ego ∈ [0.5, max_distance]   — in front of us, within range
          * |y_ego| < corridor + radius   — within our corridor

        This is intentionally simpler than the lane-aware original. The
        comparison we care about is "same planner, varying perception" — and
        the simpler corridor check is itself stable across detectors. The
        complexity we lose (lane geometry awareness) was dependent on querying
        the simulator anyway, so it would have been unfair to the perception
        models that don't get that information.
        """
        if self._agent._ignore_vehicles:
            self._brake_streak = 0
            return (False, None)
        if not self._detections:
            self._brake_streak = 0
            return (False, None)

        if max_distance is None:
            max_distance = self._agent._base_vehicle_threshold
        half_w = self._corridor_half_width

        # Get current ego speed for stationary-bypass check.
        try:
            v = self._actor.get_velocity()
            ego_speed = float((v.x ** 2 + v.y ** 2) ** 0.5)
        except Exception:
            ego_speed = 0.0
        ego_is_stopped = ego_speed < self._stopped_speed_m_s

        in_corridor = False
        for det in self._detections:
            x_ego = float(det.xy_ego[0])
            y_ego = float(det.xy_ego[1])
            if x_ego < 0.5 or x_ego > max_distance:
                continue
            # LATERAL filter — use realistic vehicle half-width (0.9m default)
            # instead of det.radius (1.5m) to stop counting parked side-cars
            # at 2.5-3m lateral as in-lane blockers.
            if abs(y_ego) > half_w + self._lateral_clearance_m:
                continue
            # STATIONARY-EGO BYPASS — when ego is essentially stopped AND
            # obstacle is far enough forward, real motion-based collision
            # risk is zero (closing speed ≈ 0 → TTC → ∞). Skip braking
            # here so ego can creep past stationary obstacles instead of
            # getting locked next to them.
            if ego_is_stopped and x_ego > self._stopped_min_forward_m:
                continue
            in_corridor = True
            break

        # Persistence filter: only commit the brake once the in-corridor
        # condition has held for N consecutive ticks. Resets immediately on
        # any clean tick, so genuine hazards still fire after N ticks of
        # consistent observation.
        if in_corridor:
            self._brake_streak += 1
        else:
            self._brake_streak = 0
        if self._brake_streak >= self._brake_persistence_ticks:
            return (True, None)
        return (False, None)
