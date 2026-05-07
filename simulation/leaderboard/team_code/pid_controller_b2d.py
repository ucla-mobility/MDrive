from collections import deque
import os
import numpy as np

class PID(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative



class PIDController(object):
    
    def __init__(self, turn_KP=0.75, turn_KI=0.75, turn_KD=0.3, turn_n=40, speed_KP=5.0, speed_KI=0.5,speed_KD=1.0, speed_n = 40,max_throttle=0.75, brake_speed=0.4,brake_ratio=1.1, clip_delta=0.25, aim_dist=4.0, angle_thresh=0.3, dist_thresh=10):
        
        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh
        # Multi-tick history for the time-based drift gate. We track the
        # most recent N values of |target_lat| (cross-track error magnitude)
        # so the override can distinguish SUSTAINED, GROWING drift (genuine
        # closed-loop drift) from a TRANSIENT large value (planned merge,
        # which converges as the ego enters the new lane).
        self._target_lat_history = deque(maxlen=12)

    def control_pid(self, waypoints, speed, target, tail_mode=False, tail_passed_terminal=False):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            speed (tensor): speedometer input
        '''

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += np.linalg.norm(
                    waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist-best_norm) > abs(self.aim_dist-norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # ── Blend selection ───────────────────────────────────────────────
        # ORIGINAL Bench2DriveZoo behaviour: binary use_target_to_aim flip
        #   if |angle_target| < |angle|: use route target
        #   elif sudden_turn condition: use route target
        #   else: use model trajectory
        # which yields w_target ∈ {0, 1}. This is the controller as the
        # UniAD/VAD authors designed it. Default ON for fair-baseline runs.
        #
        # Opt-in OFF for our research tweaks:
        #   PID_USE_BINARY_BLEND=0 → use sigmoid blend (smooth weight, less
        #     flicker on noise — see investigation/CLOSED_LOOP_SIM_FINDINGS.md).
        use_binary_blend = os.environ.get("PID_USE_BINARY_BLEND", "1").lower() in ("1", "true", "yes")
        terminal_behind_guard = bool(tail_mode and target[1] <= 0.0)

        sudden_turn = bool(
            np.abs(angle_target - angle_last) > self.angle_thresh
            and target[1] < self.dist_thresh
        )

        if use_binary_blend:
            use_target_to_aim = np.abs(angle_target) < np.abs(angle)
            if sudden_turn:
                use_target_to_aim = True
            if terminal_behind_guard:
                use_target_to_aim = False
            w_target = 1.0 if use_target_to_aim else 0.0
        else:
            mag_diff = float(np.abs(angle) - np.abs(angle_target))
            w_target = 1.0 / (1.0 + np.exp(-8.0 * mag_diff))
            w_target = max(w_target, 0.15)
            # sudden_turn: opt-in disable via PID_SUDDEN_TURN_DISABLE=1
            if sudden_turn and os.environ.get("PID_SUDDEN_TURN_DISABLE", "").lower() not in ("1", "true", "yes"):
                w_target = 1.0
            if terminal_behind_guard:
                w_target = 0.0

        # ── Open-loop drift recovery (Mode-B compensation) ───────────────
        # nuScenes-trained planners (UniAD, VAD) suffer from accumulated
        # lateral drift in closed-loop CARLA: each tick the model predicts
        # a forward trajectory assuming ego is on the centerline, but small
        # per-tick steering errors compound into 2–4 m of outward drift
        # through tight curves (we measured this on Roundabout_Navigation/3
        # ego2: 0.85 m drift at t40 → 3.25 m at t298, model running open-loop).
        #
        # SIGNATURE OF DRIFT:
        #   * model's predicted aim is mostly forward (|aim_lat| < 0.5 m),
        #     because the model thinks the ego is on the route, OR
        #   * route target is significantly off-axis in ego frame
        #     (|target_lat| > drift_lat_thresh).
        #
        # When BOTH hold, the default sigmoid blend gives the route signal
        # only ~15-20% weight (because |angle| < |angle_target| → mag_diff
        # negative → sigmoid near zero), discarding the very correction we
        # need. We override w_target to a higher floor.
        #
        # NORMAL TURNS are unaffected: the model's predicted aim swings to
        # match the upcoming turn, so |aim_lat| > drift_lat_thresh and the
        # drift override does not fire — the existing sigmoid blend handles it.
        #
        # FAIR-BASELINE CHANGE (2026-05-04): drift override is a research
        # contribution, not part of original Bench2DriveZoo. Default OFF so
        # baseline runs match the as-shipped controller. Enable per-run with
        # PID_DRIFT_OVERRIDE_ENABLE=1 (ablation/contribution studies).
        drift_override_active = False
        if os.environ.get("PID_DRIFT_OVERRIDE_ENABLE", "").lower() in ("1", "true", "yes"):
            try:
                aim_lat = float(aim[0])
                target_lat = float(target[0])
                target_fwd = float(target[1])
            except Exception:
                aim_lat = target_lat = target_fwd = 0.0
            # Default 1.5 m chosen from sweep on Roundabout_Navigation R/3 + R/5:
            #   1.5/0.7 → DS=70.99 (best of 8 configs)
            #   1.0/0.7 → DS=63.80 (prior default)
            #   0.5/0.7 → DS=55.53 (over-corrects, doubled collision rate)
            # The override is most useful as a safety net for genuine large-drift
            # events; firing too aggressively (<0.7 m threshold) causes
            # over-correction and *more* collisions, not fewer.
            # Defaults updated 2026-05-03 from faithful-sim sweep on R3/R6/HOM10/CZ3:
            # NO_GATES + override_w=0.85 + sudden_turn_disable wins all classes.
            # plan_lat_thresh=999 (disabled) and time_gate_disable=1 are now defaults.
            drift_lat_thresh = float(os.environ.get("PID_DRIFT_LAT_THRESH", "1.5"))
            model_forward_thresh = float(os.environ.get("PID_DRIFT_AIM_THRESH", "0.5"))
            override_w = float(os.environ.get("PID_DRIFT_OVERRIDE_W", "0.85"))
            # Inspect the FULL predicted plan, not only ``aim`` (which is only
            # one waypoint, picked by ``aim_dist`` near 4 m). The model often
            # plans a smooth merge or lane-change as a gradual lateral ramp:
            # ``aim`` (the first waypoint, ~3 m ahead) is near zero, but the
            # later plan waypoints have substantial lateral. If we only checked
            # ``aim_lat`` we'd misread "ramp-in" merges as "model going forward"
            # and force a route-correction that fights the planned merge.
            # Observed concretely on Highway_On-Ramp_Merge/10 ego1:
            #   aim = (-0.26, 2.67)    plan_lat_max = 1.74 m
            # Override would fire (aim_lat<0.5, target_lat>1.5) and pull the
            # ego AWAY from its planned merge → collision with merging traffic.
            # Fix: also require the model's plan_lat_max to be small (no
            # planned lateral motion). If the model's plan extends laterally
            # by more than plan_lat_thresh metres, it is intentionally moving
            # sideways and the override must not fire.
            plan_lat_thresh = float(os.environ.get("PID_DRIFT_PLAN_LAT_THRESH", "999.0"))
            try:
                plan_lat_max = max(abs(float(p[0])) for p in waypoints)
            except Exception:
                plan_lat_max = 0.0

            # ── Time-based drift gate (sustained-growing |target_lat|) ──
            # The plan_lat gate is too strict on roundabouts (model legitimately
            # plans 8-11 m of lateral motion through a curve) and too loose on
            # merges (small lateral plan but ego is laterally moving). The
            # cleanest discriminator is the TIME EVOLUTION of |target_lat|:
            #   - drift: |target_lat| grows over consecutive ticks (ego pulls
            #     further from the route as the model continues forward-only)
            #   - merge: |target_lat| shrinks as the ego enters the target lane
            #   - approaching turn: |target_lat| grows then RESETS when the
            #     waypoint pops (so window-mean stays similar)
            # Override fires when recent_window_mean(|target_lat|) is BOTH
            # large AND larger than older_window_mean(|target_lat|) by a
            # meaningful amount.
            self._target_lat_history.append(abs(target_lat))
            # Default disabled (faithful-sim sweep showed time gate over-suppresses).
            time_gate_disable = (os.environ.get("PID_DRIFT_TIME_GATE_DISABLE", "1")
                                 .lower() in ("1", "true", "yes"))
            time_gate_grow_min = float(os.environ.get("PID_DRIFT_TIME_GATE_GROW_M", "0.3"))
            sustained_growing_drift = True  # default permissive when too few ticks
            if not time_gate_disable and len(self._target_lat_history) >= 6:
                history = list(self._target_lat_history)
                # split: older half vs newer half
                half = len(history) // 2
                older_mean = sum(history[:half]) / max(1, half)
                newer_mean = sum(history[half:]) / max(1, len(history) - half)
                # SHRINKING (newer < older - small_eps) or STABLE-AT-LOW
                # cancels the drift signal.
                sustained_growing_drift = (
                    newer_mean > drift_lat_thresh
                    and newer_mean > older_mean + time_gate_grow_min
                )

            # Disable plan_lat gate (legacy) by setting threshold ≥ 999.
            plan_lat_gate_pass = (plan_lat_max < plan_lat_thresh)

            # Override fires when:
            #   - not in tail mode endpoint guard
            #   - target is ahead of ego (target_fwd > 0.5)
            #   - model's aim is mostly forward (|aim_lat| < model_forward_thresh)
            #   - the plan_lat gate passes (model not plotting big lateral)
            #   - the time gate passes (sustained growing drift)
            #   - and the route demands a meaningful lateral correction
            if (
                not terminal_behind_guard
                and target_fwd > 0.5
                and abs(aim_lat) < model_forward_thresh
                and plan_lat_gate_pass
                and sustained_growing_drift
                and abs(target_lat) > drift_lat_thresh
            ):
                if w_target < override_w:
                    w_target = override_w
                    drift_override_active = True

        angle_final = w_target * angle_target + (1.0 - w_target) * angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        # ── Hard-stop at route end (runtime optimization — default ON) ──
        # Triggers only AFTER the planner reports sticky-past-terminal
        # (route already complete; scoring has happened). Without this the
        # model keeps emitting forward trajectories indefinitely past the
        # goal, ego drives into curbs, and the sim hits the per-scenario
        # timeout — slowing eval throughput dramatically without changing
        # baseline DS. Disable with PID_TAIL_HARDSTOP_DISABLE=1.
        tail_hardstop = False
        if (
            tail_passed_terminal
            and os.environ.get("PID_TAIL_HARDSTOP_DISABLE", "").lower()
                not in ("1", "true", "yes")
        ):
            throttle = 0.0
            brake = 1.0
            steer = float(np.clip(steer * 0.2, -0.3, 0.3))  # damp residual steering
            tail_hardstop = True

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(waypoints[3].astype(np.float64)),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final),
            'delta': float(delta.astype(np.float64)),
            'tail_mode': bool(tail_mode),
            'tail_passed_terminal': bool(tail_passed_terminal),
            'terminal_behind_guard': bool(terminal_behind_guard),
            'w_target': float(w_target),
            'sudden_turn': bool(sudden_turn),
            'drift_override_active': bool(drift_override_active),
            'tail_hardstop': bool(tail_hardstop),
        }

        return steer, throttle, brake, metadata
