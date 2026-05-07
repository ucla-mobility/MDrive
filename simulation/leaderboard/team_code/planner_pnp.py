import os
from collections import deque

import numpy as np


DEBUG = int(os.environ.get("HAS_DISPLAY", 0))


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

        self.draw.ellipse((x - r, y - r, x + r, y + r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = []
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0])  # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945])  # for carla 9.10

        self.debug = Plotter(debug_size)

    def set_route(self, global_plan, gps=False):
        self.route.clear()
        route_num = len(global_plan)
        for route_id in range(route_num):
            route_tmp = deque()
            for pos, cmd in global_plan[route_id]:
                if gps:
                    pos = np.array([pos["lat"], pos["lon"]])
                    pos -= self.mean
                    pos *= self.scale
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean
                route_tmp.append((pos, cmd))
            self.route.append(route_tmp)

    def run_step(self, gps, vehicle_num):
        self.debug.clear()

        if len(self.route[vehicle_num]) == 1:
            return self.route[vehicle_num][0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route[vehicle_num])):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(
                self.route[vehicle_num][i][0] - self.route[vehicle_num][i - 1][0]
            )
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

        # Projection-based advancement: pop any waypoints the ego has already
        # passed along the route segment, regardless of perpendicular distance.
        # The position-based loop above can miss waypoints when the ego tunnels
        # through the min_distance window between planner ticks (downsampled
        # routes + skip_frames>1). Without this, the safe_ahead guard below
        # would fire indefinitely with route[1] pinned to a passed waypoint,
        # leaving target_point stuck at safe_ahead m forward (OOD-close for
        # the model). With this, route[1] always points to the next unpassed
        # waypoint along the route, restoring the model's expected target
        # distribution (~min_distance to ~hop spacing forward).
        while len(self.route[vehicle_num]) > 2:
            seg_vec = self.route[vehicle_num][1][0] - self.route[vehicle_num][0][0]
            seg_norm = np.linalg.norm(seg_vec)
            if seg_norm < 1e-6:
                break
            seg_dir = seg_vec / seg_norm
            remaining = float(np.dot(self.route[vehicle_num][1][0] - gps, seg_dir))
            if remaining > 0:
                break
            self.route[vehicle_num].popleft()

        # Guard: whenever route[1] is within safe_ahead of ego (including past
        # it), return a projected forward point.  Fires in tail mode AND
        # mid-route on tight geometry (roundabout curves, dense on-ramp
        # waypoints at high speed). Mirrors the guard in planner.py and
        # planner_b2d.py; without it, downsample_route + skip_frames lets the
        # ego tunnel past route[1] mid-route, leaving target_point pointing
        # behind the ego for the remainder of the run.
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

    def get_future_waypoints(self, vehicle_num, num=10):
        res = []
        for i in range(min(num, len(self.route[vehicle_num]))):
            res.append(
                [self.route[vehicle_num][i][0][0], self.route[vehicle_num][i][0][1], self.route[vehicle_num][i][1].value]
            )
        return res
