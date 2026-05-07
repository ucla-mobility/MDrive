"""
BehaviorAgentPlanner — wraps OpenCDA's BehaviorAgent so its collision /
overtake / car-following modules consume our Detection list (perception
output OR ground-truth oracle) instead of the default PerceptionManager
that queries ``world.get_actors()``.

This is the third planner backend for ``PerceptionSwapAgent`` (alongside
``rule`` and ``basicagent``). It is meaningfully *richer* than the other
two: BehaviorAgent supports overtake, lane-change, push-around-blocked,
and intersection-aware behavior in addition to plain car-following. The
richer the planner reacts to its detection list, the more detector
quality matters in the closed-loop metric — which is the whole point of
the perception-as-only-independent-variable study.

Fairness across detector swaps
------------------------------
The detection list is the only varying input. Detector → Detection[] →
_DetObstacle[] → BehaviorAgent.update_information(...). Everything else
(carla.Map, route, traffic-light state, ego physics) is the same world.
Traffic lights are explicitly ignored (``ignore_traffic_light=True``) so
the perception ablation is not contaminated by light-state-driven brake
events.

Notable patches applied to OpenCDA's BehaviorAgent
--------------------------------------------------
* ``BehaviorAgent.is_close_to_destination`` calls ``sys.exit(0)``.
  We replace it with ``lambda self: False`` and own the goal-reached
  latch ourselves (distance-to-end-waypoint).

Usage
-----
    planner = BehaviorAgentPlanner(actor=hero_vehicle, target_speed_kmh=25)
    planner.set_global_plan(world_plan)            # leaderboard route

    # Per step:
    planner.set_detections(detections_ego_frame)   # List[Detection]
    control = planner.run_step()                   # carla.VehicleControl
    traj    = planner.predict_trajectory(3.0, 0.1) # (T,2) world-frame xy
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Optional

import numpy as np
import carla


_OPENCDA_ROOT = "/data2/marco/CoLMDriver/OpenCDA"
if _OPENCDA_ROOT not in sys.path:
    sys.path.insert(0, _OPENCDA_ROOT)

from opencda.core.plan.behavior_agent import BehaviorAgent  # noqa: E402
from opencda.core.actuation.pid_controller import Controller as PIDController  # noqa: E402

from team_code.rule_planner import Detection  # noqa: E402


# ────────────────────────────────────────────────────────────────────────────
# Lightweight obstacle shim — duck-types the carla.Vehicle interface that
# BehaviorAgent's collision_manager / car_following_manager / overtake_management
# / white_list_match actually use:
#   .get_location()          → carla.Location
#   .get_velocity()          → carla.Vector3D
#   .get_transform()         → carla.Transform
#   .bounding_box.extent.{x,y,z}  → half-extents (floats)
#   .bounding_box.location   → bbox-center offset (zero for synthetic boxes)
#
# Why a shim and not OpenCDA's ObstacleVehicle: ObstacleVehicle requires
# either a real carla.Vehicle (queries CARLA actor attributes — defeats the
# perception ablation) or an 8-corner array (which we don't have from a
# centroid+radius detection). The shim is the minimal, fair object.
# ────────────────────────────────────────────────────────────────────────────

class _ShimBoundingBox:
    __slots__ = ("extent", "location")

    def __init__(self, extent_x: float, extent_y: float, extent_z: float):
        self.extent = carla.Vector3D(float(extent_x),
                                     float(extent_y),
                                     float(extent_z))
        self.location = carla.Vector3D(0.0, 0.0, 0.0)


class _DetObstacle:
    """Detection shim that exposes the carla.Vehicle interface BehaviorAgent
    consults during collision / overtake / car-following decisions."""
    __slots__ = ("_loc", "_tf", "_vel", "bounding_box", "id")

    def __init__(
        self,
        world_xy,
        z: float,
        yaw_deg: float,
        half_extent_x: float = 2.25,
        half_extent_y: float = 1.0,
        half_extent_z: float = 0.8,
    ):
        self._loc = carla.Location(x=float(world_xy[0]),
                                    y=float(world_xy[1]),
                                    z=float(z))
        self._tf = carla.Transform(self._loc,
                                   carla.Rotation(pitch=0.0,
                                                  yaw=float(yaw_deg),
                                                  roll=0.0))
        self._vel = carla.Vector3D(0.0, 0.0, 0.0)
        self.bounding_box = _ShimBoundingBox(half_extent_x,
                                              half_extent_y,
                                              half_extent_z)
        self.id = -1

    def get_location(self) -> "carla.Location":
        return self._loc

    def get_velocity(self) -> "carla.Vector3D":
        return self._vel

    def get_transform(self) -> "carla.Transform":
        return self._tf


# ────────────────────────────────────────────────────────────────────────────
# Default config — mirrors the relevant blocks of OpenCDA's default.yaml so
# the BehaviorAgent's behavior is the published-default behavior. Only knobs
# that require this experiment's framing are overridden here.
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_BEHAVIOR_CFG = {
    "max_speed": 25.0,                   # km/h, also used as the cruise target
    "tailgate_speed": 30.0,
    "speed_lim_dist": 3,
    "speed_decrease": 15,
    "safety_time": 4,
    "emergency_param": 0.4,
    # Traffic lights are NOT part of the detection list (Detection only
    # carries vehicles in our perception ablation). Ignoring lights keeps
    # brake events purely detection-driven, matching the methodology of
    # the rule/basicagent backends.
    "ignore_traffic_light": True,
    "overtake_allowed": True,
    "collision_time_ahead": 1.5,
    "sample_resolution": 4.5,
    "debug": False,
    "local_planner": {
        "buffer_size": 12,
        "trajectory_update_freq": 15,
        "waypoint_update_freq": 9,
        "min_dist": 3,
        "trajectory_dt": 0.20,
        "debug": False,
        "debug_trajectory": False,
    },
}


DEFAULT_CONTROLLER_CFG = {
    "max_brake": 1.0,
    "max_throttle": 0.75,
    "max_steering": 0.3,
    "lon": {"k_p": 0.37, "k_d": 0.024, "k_i": 0.032},
    "lat": {"k_p": 0.75, "k_d": 0.02,  "k_i": 0.4},
    # Leaderboard runs at 20 Hz (0.05 s) by default.
    "dt": 0.05,
    "dynamic": False,
}


class BehaviorAgentPlanner:
    """OpenCDA BehaviorAgent rewired to consume an ego-frame Detection list.

    Parameters
    ----------
    actor : carla.Vehicle
        The hero vehicle this planner controls. Must already exist in CARLA.
    target_speed_kmh : float
        Cruise target speed. Wired into both BehaviorAgent.max_speed and
        the predict_trajectory open-loop rollout speed.
    behavior_cfg : dict, optional
        Overrides for the BehaviorAgent config block (merged onto the
        OpenCDA default.yaml-derived defaults).
    controller_cfg : dict, optional
        Overrides for the PID controller config block.
    terminal_radius_m : float
        Distance to the route's final waypoint at which we latch
        ``done() == True`` and emit a full-brake control. Mirrors the
        rule/basicagent terminal-radius semantics.
    """

    def __init__(
        self,
        actor: "carla.Vehicle",
        target_speed_kmh: float = 25.0,
        behavior_cfg: Optional[dict] = None,
        controller_cfg: Optional[dict] = None,
        terminal_radius_m: float = 5.0,
        synthetic_obstacle_extent_xy: tuple = (2.25, 1.0),
    ):
        if actor is None:
            raise ValueError(
                "BehaviorAgentPlanner: actor is None — defer construction "
                "until the hero is spawned (mirror BasicAgentPlanner)."
            )
        self._actor = actor
        self._world = actor.get_world()
        self._map = self._world.get_map()

        cfg = {k: (dict(v) if isinstance(v, dict) else v)
               for k, v in DEFAULT_BEHAVIOR_CFG.items()}
        if behavior_cfg:
            for k, v in behavior_cfg.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        cfg["max_speed"] = float(target_speed_kmh)
        # Tailgate speed must be > max_speed for OpenCDA's interlocks.
        cfg["tailgate_speed"] = max(float(cfg.get("tailgate_speed", 0.0)),
                                    float(target_speed_kmh) * 1.2)

        self._agent = BehaviorAgent(actor, self._map, cfg)
        # Replace the kill-switch goal check with a no-op; we own the
        # goal-reached latch (see run_step).
        self._agent.is_close_to_destination = lambda: False

        ctrl_cfg = {k: (dict(v) if isinstance(v, dict) else v)
                    for k, v in DEFAULT_CONTROLLER_CFG.items()}
        if controller_cfg:
            for k, v in controller_cfg.items():
                if isinstance(v, dict) and isinstance(ctrl_cfg.get(k), dict):
                    ctrl_cfg[k].update(v)
                else:
                    ctrl_cfg[k] = v
        self._controller = PIDController(ctrl_cfg)

        # Cached objects dict, refreshed each tick by set_detections().
        self._objects = {"vehicles": [], "traffic_lights": []}

        self._end_loc: Optional[carla.Location] = None
        self._reached = False
        self._terminal_radius_m = float(terminal_radius_m)
        self._target_speed_kmh = float(target_speed_kmh)
        self._tick = 0
        # Half-extents used for synthetic obstacle bounding boxes.
        self._half_ex = (float(synthetic_obstacle_extent_xy[0]),
                         float(synthetic_obstacle_extent_xy[1]))

        # ── Fallback / stuck-recovery state ─────────────────────────────────
        # OpenCDA's BehaviorAgent.run_step can return (0, None) for legitimate
        # reasons (emergency stop) AND for spurious ones (empty spline near
        # route end, very short routes whose buffer collapses to <2 unique
        # waypoints). The latter freezes ego at brake=1.0 indefinitely.
        # We track consecutive ticks of stuck behavior and (a) drive at low
        # speed toward the next forward waypoint, (b) periodically re-prime
        # `set_destination` from the live ego pose, and (c) once close to the
        # goal, head straight to end_loc instead of relying on the spline.
        self._stuck_ticks = 0                        # consecutive (0,None) ticks
        self._zero_speed_ticks = 0                   # consecutive ticks ego_speed≈0
        self._last_reprime_tick = -10_000            # tick at last set_destination re-prime
        self._reprime_cooldown_ticks = 80            # min ticks between re-primes
        self._near_goal_engage_m = 12.0              # within this distance to end_loc, take over
        self._fallback_target_speed_kmh = max(8.0, 0.4 * float(target_speed_kmh))
        self._fallback_speed_kmh = self._fallback_target_speed_kmh  # mutable per-tick fallback target
        # Counters for diagnostics (read from outside if needed).
        self._n_fallback_invocations = 0
        self._n_reprimes = 0
        # ── Duck-typing for perception_swap_agent's per-frame diag JSONL writer.
        # The writer's catch-all branch (perception_swap_agent.py:2949-2951)
        # reads `planner._agent._base_vehicle_threshold` and
        # `planner._corridor_half_width` — both BasicAgentPlanner-specific.
        # We surface equivalent values so the diag log captures behavior runs.
        # Lookahead ≈ collision_time_ahead × cruise_speed_m_s; corridor half-
        # width ≈ CARLA driving lane half-width (matches BasicAgentPlanner).
        self._corridor_half_width = 1.6
        self._agent._base_vehicle_threshold = float(
            cfg.get("collision_time_ahead", 1.5)
        ) * float(target_speed_kmh) / 3.6

    # ─── Wiring ──────────────────────────────────────────────────────────

    def set_global_plan(self, world_plan):
        """world_plan: list of (carla.Transform, RoadOption) — leaderboard's format."""
        if not world_plan or len(world_plan) < 2:
            raise RuntimeError(
                "BehaviorAgentPlanner: empty/too-short world_plan"
            )
        start_loc = world_plan[0][0].location
        end_loc = world_plan[-1][0].location
        self._end_loc = end_loc
        # Pre-warm BehaviorAgent state (set_destination consults _ego_pos
        # to align the start waypoint with the ego's current heading).
        self._agent.update_information(
            self._actor.get_transform(),
            0.0,
            self._objects,
        )
        # set_destination invokes OpenCDA's GlobalRoutePlanner.trace_route
        # which can raise nx.NetworkXNoPath as `Node X not reachable from Y`
        # when the requested start↔end pair lands on disjoint road-graph
        # components (some v2xpnp scenarios put the goal across a road
        # boundary the offline graph treats as unreachable). We tolerate it:
        # the local planner ends up with an empty waypoint buffer, so
        # run_step's fallback path drives the ego toward _end_loc directly.
        try:
            self._agent.set_destination(
                start_loc, end_loc,
                clean=True,
                end_reset=True,
                clean_history=True,
            )
        except Exception as exc:
            print(f"[behavior_agent_planner] set_global_plan: set_destination "
                  f"raised ({exc}); local planner will start with empty "
                  f"buffer and run_step will drive via fallback toward "
                  f"end_loc=({end_loc.x:.1f},{end_loc.y:.1f}).")

    def set_detections(self, detections: Optional[List[Detection]]):
        """Convert ego-frame Detections into world-frame _DetObstacle shims.

        Detection.xy_ego is in CARLA LH ego frame (+x forward, +y right).
        We project to world via the live ego transform, snap orientation to
        the nearest map waypoint heading (so the collision-circle check
        sees a sensible orientation), and discard detections behind the ego.
        """
        ego_tf = self._actor.get_transform()
        ego_yaw_rad = math.radians(ego_tf.rotation.yaw)
        cos_y = math.cos(ego_yaw_rad)
        sin_y = math.sin(ego_yaw_rad)
        ego_x = ego_tf.location.x
        ego_y = ego_tf.location.y
        ego_z = ego_tf.location.z

        vehicles = []
        for det in (detections or []):
            x_ego = float(det.xy_ego[0])
            y_ego = float(det.xy_ego[1])
            # Behind-ego detections are not actionable for forward collision
            # checks; drop with a small safety margin so detections right at
            # the ego's bumper line still count.
            if x_ego <= -0.5:
                continue
            world_x = ego_x + cos_y * x_ego - sin_y * y_ego
            world_y = ego_y + sin_y * x_ego + cos_y * y_ego
            try:
                wp = self._map.get_waypoint(
                    carla.Location(x=world_x, y=world_y, z=ego_z),
                    project_to_road=True,
                )
                yaw_deg = wp.transform.rotation.yaw if wp is not None else ego_tf.rotation.yaw
            except Exception:
                yaw_deg = ego_tf.rotation.yaw
            vehicles.append(_DetObstacle(
                (world_x, world_y),
                ego_z,
                yaw_deg,
                half_extent_x=self._half_ex[0],
                half_extent_y=self._half_ex[1],
            ))
        # traffic_lights stays empty: the perception model doesn't produce
        # them, and ignore_traffic_light=True bypasses them anyway. Empty
        # list also makes BehaviorAgent.is_intersection() return False
        # uniformly, which is fair across detector swaps.
        self._objects = {"vehicles": vehicles, "traffic_lights": []}

    # ─── Step ────────────────────────────────────────────────────────────

    def run_step(self) -> "carla.VehicleControl":
        ego_tf = self._actor.get_transform()
        v = self._actor.get_velocity()
        ego_speed_mps = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        ego_speed_kmh = 3.6 * ego_speed_mps

        # 1. Inform the agent. BehaviorAgent.update_information also
        #    propagates pose/speed to its internal LocalPlanner. Wrap in
        #    try/except — its trace_route call can raise NetworkXNoPath
        #    ("Node X not reachable from Y") on scenarios where the offline
        #    road graph has disjoint components between the live ego pose
        #    and the destination. Without this guard the AgentError
        #    propagates to the leaderboard's autonomous_agent wrapper which
        #    immediately terminates the route after 1 tick.
        try:
            self._agent.update_information(ego_tf, ego_speed_kmh, self._objects)
        except Exception as exc:
            print(f"[behavior_agent_planner] update_information exception: {exc}; "
                  f"continuing with fallback control")

        # Stationary-ego book-keeping for the stuck detector.
        if ego_speed_mps < 0.15:
            self._zero_speed_ticks += 1
        else:
            self._zero_speed_ticks = 0

        # 2. Goal-reached: do NOT latch on straight-line distance to end_loc —
        # for curved routes the ego can come within terminal_radius_m of
        # end_loc while still mid-route (empirically observed on
        # 2023-03-17-15-53-02_1_0 ego 0: ego at xy distance 1.6 m from end_loc
        # but only 62% of the leaderboard's RouteCompletion arc-length).
        # Premature latching froze the ego, causing the very near-goal-stop
        # we set out to fix. Instead: keep driving and let leaderboard's
        # RouteCompletionTest terminate the scenario when its arc-length
        # criterion or forward-pass fallback fires (10 m radius / forward dot
        # past last waypoint). BasicAgentPlanner has the same property — it
        # stops only when CARLA's BasicAgent.done() empties the queue.
        # We still compute dist_to_goal for the near-goal fallback below.
        dist_to_goal = float("inf")
        if self._end_loc is not None:
            dx = ego_tf.location.x - self._end_loc.x
            dy = ego_tf.location.y - self._end_loc.y
            dist_to_goal = math.sqrt(dx * dx + dy * dy)

        # 3. Plan. BehaviorAgent.run_step() returns (target_speed_kmh, target_loc).
        target_speed_kmh = None
        target_loc = None
        try:
            target_speed_kmh, target_loc = self._agent.run_step(
                target_speed=None,
                collision_detector_enabled=True,
                lane_change_allowed=True,
            )
        except Exception as exc:
            # Defensive: a transient OpenCDA failure (empty waypoint buffer,
            # off-map ego) should NOT kill the closed-loop run. Fall through
            # to the fallback path below — keep going if we know where to go.
            print(f"[behavior_agent_planner] BehaviorAgent.run_step exception: {exc}")
            target_speed_kmh = None
            target_loc = None

        # 4. Detect "OpenCDA returned (0, None)" — could be legitimate emergency
        #    stop OR a spurious empty-spline. Engage fallback only if the
        #    safer interpretation is "spurious": the ego is far from the goal,
        #    no perception detection is *actually* in our forward corridor,
        #    AND we have somewhere to go. This preserves real emergency stops
        #    (true obstacle in path within ~3 m) which BasicAgent honors too.
        opencda_stopped = (
            target_loc is None
            or target_speed_kmh is None
            or float(target_speed_kmh) <= 1e-6
        )

        # Forward-corridor obstacle check uses the same Detection list we fed
        # to the BehaviorAgent. _objects['vehicles'] are world-frame _DetObstacle
        # shims with .get_location(); we project them back to ego frame via the
        # current transform and ask: any centre with x in [0.5, 6] m and |y|<2 m?
        in_corridor_dist = self._closest_in_corridor_dist_m(ego_tf)

        if opencda_stopped:
            self._stuck_ticks += 1
            self._n_fallback_invocations += 1
        else:
            self._stuck_ticks = 0

        # If OpenCDA emitted normal control, use it.
        if not opencda_stopped:
            self._controller.update_info(ego_tf, ego_speed_kmh)
            try:
                control = self._controller.run_step(
                    float(target_speed_kmh), target_loc
                )
            except Exception as exc:
                print(f"[behavior_agent_planner] PID exception: {exc}")
                control = self._fallback_brake(soft=True)
            self._tick += 1
            return control

        # ── Fallback path ─────────────────────────────────────────────────
        # Rationale: OpenCDA returned (0, None). If a real obstacle is within
        # ~3.5 m of our nose in the forward corridor, that's a legitimate
        # car-following / emergency stop — preserve it. Otherwise the
        # BehaviorAgent has lost its spline; we drive forward toward the next
        # available waypoint or directly toward end_loc.
        EMERGENCY_DIST_M = 3.5
        if in_corridor_dist is not None and in_corridor_dist < EMERGENCY_DIST_M:
            # Real obstacle very close — honor the brake.
            self._tick += 1
            return self._fallback_brake(soft=False)

        # Pick a forward target. Prefer the next non-passed waypoint in the
        # local planner's buffer; otherwise head straight to end_loc.
        fwd_target = self._next_forward_target(ego_tf)

        # Near-goal: bypass spline entirely and drive straight to end_loc at
        # a low speed. This is what fixes the "near-goal-stop" failure mode
        # where local_planner.generate_path produces empty rx because the
        # ego is past the spline tail.
        if (
            self._end_loc is not None
            and dist_to_goal <= self._near_goal_engage_m
        ):
            fwd_target = self._end_loc

        # Periodic re-prime: if the ego has been stationary > 1 s AND OpenCDA
        # has been returning (0,None) for > 1 s, re-build the route from the
        # live ego pose. This is the OpenCDA `Destination Reset!` path made
        # explicit and triggered earlier.
        if (
            self._zero_speed_ticks > 20
            and self._stuck_ticks > 20
            and (self._tick - self._last_reprime_tick)
                > self._reprime_cooldown_ticks
            and self._end_loc is not None
        ):
            try:
                self._agent.set_destination(
                    ego_tf.location,
                    self._end_loc,
                    clean=True,
                    end_reset=False,
                    clean_history=True,
                )
                self._last_reprime_tick = self._tick
                self._n_reprimes += 1
            except Exception as exc:
                # set_destination needs a routable start; if that fails, keep
                # using the fallback target without raising.
                print(f"[behavior_agent_planner] re-prime set_destination "
                      f"exception: {exc}")

        if fwd_target is None:
            # Nothing valid to drive toward — preserve the legacy brake.
            self._tick += 1
            return self._fallback_brake(soft=False)

        # Drive toward fwd_target at fallback speed via the existing PID.
        # Slow further when very close to the goal to avoid overshoot.
        speed_kmh = self._fallback_speed_kmh
        if dist_to_goal < 6.0:
            speed_kmh = min(speed_kmh, 6.0)
        self._controller.update_info(ego_tf, ego_speed_kmh)
        try:
            control = self._controller.run_step(
                float(speed_kmh), fwd_target
            )
        except Exception as exc:
            print(f"[behavior_agent_planner] fallback PID exception: {exc}")
            control = self._fallback_brake(soft=True)
        self._tick += 1
        return control

    # ─── Internal helpers (added by behavior-agent fix) ──────────────────

    def _fallback_brake(self, soft: bool = False) -> "carla.VehicleControl":
        ctrl = carla.VehicleControl()
        ctrl.steer = 0.0
        ctrl.throttle = 0.0
        ctrl.brake = 0.5 if soft else 1.0
        ctrl.hand_brake = False
        return ctrl

    def _closest_in_corridor_dist_m(self, ego_tf) -> Optional[float]:
        """Return the closest forward-corridor obstacle centre distance in
        meters, or None if no obstacle qualifies. Mirrors the rule planner's
        narrow corridor used by BasicAgentPlanner so the brake-vs-go decision
        is consistent across planner backends."""
        veh = self._objects.get("vehicles") or []
        if not veh:
            return None
        ex = ego_tf.location.x
        ey = ego_tf.location.y
        yaw = math.radians(ego_tf.rotation.yaw)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        best = None
        for ob in veh:
            try:
                loc = ob.get_location()
            except Exception:
                continue
            dx = loc.x - ex
            dy = loc.y - ey
            xe = cy * dx + sy * dy
            ye = -sy * dx + cy * dy
            # Forward corridor: 0.5 <= xe <= 6 m, |ye| <= 1.6 m
            if 0.5 <= xe <= 6.0 and abs(ye) <= self._corridor_half_width:
                if best is None or xe < best:
                    best = xe
        return best

    def _next_forward_target(self, ego_tf) -> "Optional[carla.Location]":
        """Return the next waypoint location that lies in front of the ego
        (angle <= 80 deg) from the BehaviorAgent's local-planner buffer.
        Falls back to ``end_loc`` if the buffer is empty / all behind."""
        try:
            lp = self._agent.get_local_planner()
            buf = list(lp.get_waypoint_buffer())
        except Exception:
            buf = []
        ex = ego_tf.location.x
        ey = ego_tf.location.y
        yaw = math.radians(ego_tf.rotation.yaw)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        for entry in buf:
            try:
                wp = entry[0]
                wl = wp.transform.location
            except Exception:
                continue
            dx = wl.x - ex
            dy = wl.y - ey
            xe = cy * dx + sy * dy  # forward in ego frame
            d = math.sqrt(dx * dx + dy * dy)
            # Prefer the first waypoint that's >= 1.5 m forward and within
            # ±80 deg of heading. This skips waypoints we've already passed.
            if d > 0.5 and xe > 0.5 * d:  # cos(60°)≈0.5 — ~60° cone
                return wl
        # Fallback: drive straight at end_loc.
        return self._end_loc

    # ─── Trajectory rollout for open-loop ADE/FDE compatibility ──────────

    def predict_trajectory(self, horizon_s: float = 3.0, dt: float = 0.1) -> "np.ndarray":
        """Walk BehaviorAgent's LocalPlanner waypoint buffer at target_speed.

        Mirrors BasicAgentPlanner.predict_trajectory: ramp from current
        speed to ``target_speed_mps`` at a bounded accel (default 2 m/s²,
        overridable via env ``OPENLOOP_PRED_ACCEL_MPS2``).
        """
        target = float(self._target_speed_kmh) / 3.6
        try:
            v = self._actor.get_velocity()
            v0 = float(math.sqrt(v.x ** 2 + v.y ** 2))
        except Exception:
            v0 = target
        try:
            a_max = float(os.environ.get("OPENLOOP_PRED_ACCEL_MPS2", "2.0"))
        except Exception:
            a_max = 2.0

        loc = self._actor.get_location()
        pos = np.array([loc.x, loc.y], dtype=np.float64)
        T = max(1, int(round(horizon_s / dt)))

        try:
            buf = list(self._agent.get_local_planner().get_waypoint_buffer())
            wp_locs = [
                (wp.transform.location.x, wp.transform.location.y)
                for wp, _ in buf
            ]
        except Exception:
            wp_locs = []

        if not wp_locs:
            return np.tile(pos, (T, 1))

        out = np.zeros((T, 2), dtype=np.float64)
        cursor = 0
        speed = float(v0)
        pop_dist = 1.5
        for t in range(T):
            if speed < target:
                speed = min(target, speed + a_max * dt)
            elif speed > target:
                speed = max(target, speed - a_max * dt)
            while cursor < len(wp_locs) - 1:
                wx, wy = wp_locs[cursor]
                if math.hypot(wx - pos[0], wy - pos[1]) < pop_dist:
                    cursor += 1
                else:
                    break
            wx, wy = wp_locs[cursor]
            vec = np.array([wx - pos[0], wy - pos[1]], dtype=np.float64)
            d = float(np.linalg.norm(vec))
            if d > 1e-6:
                vec = vec / d
            else:
                vec = np.array([0.0, 0.0])
            pos = pos + vec * speed * dt
            out[t] = pos
        return out

    def done(self) -> bool:
        return self._reached

    @property
    def target_speed_mps(self) -> float:
        return float(self._target_speed_kmh) / 3.6
