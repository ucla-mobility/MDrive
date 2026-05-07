import os
from collections import deque

import numpy as np
import math
EARTH_RADIUS_EQUA = 6378137.0


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256, lat_ref=42.0, lon_ref=2.0):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0]) # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10

        self.debug = Plotter(debug_size)
        # self.lat_ref, self.lon_ref = self._get_latlon_ref()
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

        # ── Robust route-end detection state ────────────────────────────
        # Cached at set_route() time so that no later popleft() can corrupt
        # them. Per-vehicle (indexed by vehicle_num).
        self.route_end_pos = []         # list[np.ndarray (2,)] — final XY
        self.route_end_dir = []         # list[np.ndarray (2,)] — unit vec, average of last K=3 segments
        # Per-vehicle live flags. Sticky semantics: tail_passed_terminal,
        # once True, never goes False back (a completed route stays
        # completed even if the ego happens to drift back in the next tick).
        self.tail_mode = False                 # ego is "near" the end
        self.tail_passed_terminal = False      # ego is past the end (sticky)
        self._tail_mode_per_ego = []
        self._tail_passed_per_ego = []
        # Tunable thresholds — env var overrides for ablation.
        # tail_mode triggers when ego is within this distance of the end
        # OR within this signed-projection distance of past-the-end.
        self.tail_mode_radius = float(os.environ.get(
            "PLANNER_TAIL_MODE_RADIUS_M", "5.0"))
        # tail_passed_terminal triggers when signed forward projection past
        # end_pos exceeds this. Sticky once True.
        self.tail_passed_buffer = float(os.environ.get(
            "PLANNER_TAIL_PASSED_BUFFER_M", "1.0"))
        # tail_passed also requires that ego was at some point CLOSE to the
        # end (within this radius) — guards against false positives where a
        # severely off-route ego happens to project past the end laterally.
        self.tail_passed_proximity_required_m = float(os.environ.get(
            "PLANNER_TAIL_PASSED_PROXIMITY_M", "10.0"))
        self._tail_was_close = []  # per-vehicle bool: was ever within proximity

    def set_route(self, global_plan, gps=False, global_plan_world = None):
        self.route.clear()
        # Reset per-vehicle end-detection state — fresh per scenario.
        self.route_end_pos = []
        self.route_end_dir = []
        self._tail_mode_per_ego = []
        self._tail_passed_per_ego = []
        self._tail_was_close = []

        route_num = len(global_plan)
        for route_id in range(route_num):
            route_tmp = deque()
            if global_plan_world:
                for (pos, cmd), (pos_word, _ )in zip(global_plan[route_id], global_plan_world[route_id]):
                    if gps:
                        pos = self.gps_to_location(np.array([pos['lat'], pos['lon']]))
                        # pos -= self.mean
                        # pos *= self.scale
                    else:
                        pos = np.array([pos.location.x, pos.location.y])
                        # pos -= self.mean

                    route_tmp.append((pos, cmd, pos_word))
            else:
                for pos, cmd in global_plan[route_id]:
                    if gps:
                        pos = self.gps_to_location(np.array([pos['lat'], pos['lon']]))
                        # pos -= self.mean
                        # pos *= self.scale
                    else:
                        pos = np.array([pos.location.x, pos.location.y])
                        # pos -= self.mean

                    route_tmp.append((pos, cmd))
            self.route.append(route_tmp)

            # Cache end_pos and end_dir for this ego. We use the AVERAGE
            # direction of the last K segments rather than only the very
            # last one, because the very last segment is often short
            # (sub-metre) due to the route alignment terminating on a
            # waypoint that's almost duplicate of the prior one — that
            # gives a noisy unit vector.
            #
            # For routes with <2 waypoints (degenerate), end_pos is the
            # only point and end_dir defaults to +x; tail detection then
            # falls back to pure distance.
            if len(route_tmp) >= 2:
                last_pts = list(route_tmp)
                end_pos = np.array(last_pts[-1][0], dtype=float)
                K = min(3, len(last_pts) - 1)
                samples = []
                for i in range(len(last_pts) - K, len(last_pts)):
                    seg = np.array(last_pts[i][0], dtype=float) - np.array(last_pts[i-1][0], dtype=float)
                    sn = float(np.linalg.norm(seg))
                    if sn > 0.05:  # skip near-duplicate waypoints
                        samples.append(seg / sn)
                if samples:
                    avg = np.mean(np.stack(samples, axis=0), axis=0)
                    an = float(np.linalg.norm(avg))
                    end_dir = avg / an if an > 0.05 else np.array([1.0, 0.0])
                else:
                    end_dir = np.array([1.0, 0.0])
                self.route_end_pos.append(end_pos)
                self.route_end_dir.append(end_dir)
            elif len(route_tmp) == 1:
                self.route_end_pos.append(np.array(route_tmp[0][0], dtype=float))
                self.route_end_dir.append(np.array([1.0, 0.0]))
            else:
                self.route_end_pos.append(np.array([0.0, 0.0]))
                self.route_end_dir.append(np.array([1.0, 0.0]))
            self._tail_mode_per_ego.append(False)
            self._tail_passed_per_ego.append(False)
            self._tail_was_close.append(False)


    def _update_tail_flags(self, gps, vehicle_num):
        """Compute (tail_mode, tail_passed_terminal) for ``vehicle_num`` from
        the cached end_pos/end_dir and the current ego ``gps``. Updates
        per-vehicle state and the public ``self.tail_mode`` /
        ``self.tail_passed_terminal`` attributes (which the agents read).

        Stickiness: tail_passed_terminal can flip False→True; once True it
        STAYS True. A completed route stays completed even if the ego happens
        to drift back across the end direction.
        """
        if vehicle_num >= len(self.route_end_pos):
            self.tail_mode = False
            self.tail_passed_terminal = False
            return
        end_pos = self.route_end_pos[vehicle_num]
        end_dir = self.route_end_dir[vehicle_num]
        ego = np.asarray(gps, dtype=float)

        delta = ego - end_pos
        dist_to_end = float(np.linalg.norm(delta))
        signed_past = float(np.dot(delta, end_dir))   # >0 = past end in route direction

        # Track whether ego was ever close to the end. Required for
        # passed_terminal to fire — guards against severely off-route egos
        # that happen to project past the end laterally.
        if dist_to_end < self.tail_passed_proximity_required_m:
            self._tail_was_close[vehicle_num] = True

        # tail_mode: near the end (approaching or just past)
        in_tail_mode = (
            dist_to_end < self.tail_mode_radius
            or signed_past > -self.tail_mode_radius
        )
        # passed_terminal: clearly past, AND we got close to the end at
        # some point (sticky).
        new_passed = (
            signed_past > self.tail_passed_buffer
            and self._tail_was_close[vehicle_num]
        )
        if new_passed:
            self._tail_passed_per_ego[vehicle_num] = True
        # Once sticky-True, stay True.
        passed = self._tail_passed_per_ego[vehicle_num]

        self._tail_mode_per_ego[vehicle_num] = bool(in_tail_mode or passed)
        # Public attributes: agents read these via getattr
        self.tail_mode = self._tail_mode_per_ego[vehicle_num]
        self.tail_passed_terminal = passed


    def run_step(self, gps, vehicle_num):
        self.debug.clear()

        # Update route-end detection FIRST so the agents see the correct
        # tail_mode / tail_passed_terminal values for this tick. Robust
        # against popleft (uses cached end_pos/end_dir) and against
        # transient drift-back (passed_terminal is sticky).
        self._update_tail_flags(gps, vehicle_num)

        if len(self.route[vehicle_num]) == 1:
            return self.route[vehicle_num][0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route[vehicle_num])):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[vehicle_num][i][0] - self.route[vehicle_num][i-1][0])
            distance = np.linalg.norm(self.route[vehicle_num][i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[vehicle_num][i][1].value == 4)
            b = 255
            self.debug.dot(gps, self.route[vehicle_num][i][0], (r, g, b))

        for _ in range(to_pop):
            if len(self.route[vehicle_num]) > 2:
                self.route[vehicle_num].popleft()

        self.debug.dot(gps, self.route[vehicle_num][0][0], (0, 255, 0))
        self.debug.dot(gps, self.route[vehicle_num][1][0], (255, 0, 0))
        self.debug.dot(gps, gps, (0, 0, 255))
        self.debug.show()

        # Guard: whenever route[1] is within safe_ahead of ego (including past
        # it), return a projected forward point.  Fires in tail mode AND
        # mid-route on tight geometry (roundabout curves, dense on-ramp
        # waypoints at high speed).
        route = self.route[vehicle_num]
        seg_vec = route[1][0] - route[0][0]
        seg_norm = np.linalg.norm(seg_vec)
        if seg_norm > 1e-6:
            seg_dir = seg_vec / seg_norm
            safe_ahead = max(1.0, 0.5 * self.min_distance)
            remaining = float(np.dot(route[1][0] - gps, seg_dir))
            if remaining <= safe_ahead:
                safe_pos = gps + seg_dir * safe_ahead
                return (safe_pos, route[1][1])

        return self.route[vehicle_num][1]
    
    def gps_to_location(self, gps):
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])