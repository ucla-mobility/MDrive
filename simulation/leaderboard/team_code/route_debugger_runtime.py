import json
import datetime
from pathlib import Path

import numpy as np


class RouteRuntimeDebugger:
    """Collects detailed runtime diagnostics for route following."""

    def __init__(self, base_dir: str, route_tag: str, session_tag: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_session = session_tag.replace(" ", "_")
        self.base_dir = Path(base_dir) / f"{route_tag}_{safe_session}_{timestamp}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.plan_entries = []
        self.plan_local_points = []
        self.tick_records = []
        self.actual_positions = []
        self.distances = []
        self.centerline_distances = []
        self.command_histogram = {}
        self.divergence_events = []
        self.divergence_threshold = 4.0
        self._last_over_threshold = False
        self.reference = {}
        self._finalized = False

    def set_reference(self, lat_ref: float, lon_ref: float):
        self.reference = {"lat_ref": float(lat_ref), "lon_ref": float(lon_ref)}

    def capture_plan(self, global_plan, global_plan_world, planner_route_snapshot):
        plan_payload = []
        for idx, entry in enumerate(global_plan):
            gps_info, command = self._unpack_plan_entry(entry)
            world_dict = None
            if global_plan_world and idx < len(global_plan_world):
                transform = global_plan_world[idx][0]
                world_dict = {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                    "yaw": float(transform.rotation.yaw),
                }
            plan_payload.append(
                {
                    "idx": idx,
                    "gps": gps_info,
                    "world": world_dict,
                    "command": self._command_name(command),
                }
            )
        local_snapshot = []
        for idx, route_entry in enumerate(planner_route_snapshot):
            local_xy = route_entry[0].tolist() if hasattr(route_entry[0], "tolist") else list(route_entry[0])
            cmd = route_entry[1] if len(route_entry) > 1 else None
            world_dict = None
            if len(route_entry) > 2 and route_entry[2] is not None:
                transform = route_entry[2]
                world_dict = {
                    "x": float(transform.location.x),
                    "y": float(transform.location.y),
                    "z": float(transform.location.z),
                    "yaw": float(transform.rotation.yaw),
                }
            local_snapshot.append(
                {"idx": idx, "local_xy": local_xy, "command": self._command_name(cmd), "world": world_dict}
            )

        payload = {
            "reference": self.reference,
            "global_plan": plan_payload,
            "planner_local_snapshot": local_snapshot,
        }
        (self.base_dir / "global_plan.json").write_text(json.dumps(payload, indent=2))
        self.plan_entries = plan_payload
        self.plan_local_points = local_snapshot

    def log_tick(
        self,
        *,
        ego_id: int,
        tick_index: int,
        timestamp: float,
        gps,
        local_pos,
        speed: float,
        compass: float,
        near_node,
        near_command,
        near_world=None,
        extra_fields=None,
    ):
        gps_list = [float(g) for g in gps]
        pos_list = self._to_list(local_pos)
        waypoint = self._to_list(near_node)
        dist = float(np.linalg.norm(np.array(pos_list) - np.array(waypoint)))
        cmd_name = self._command_name(near_command)
        record = {
            "ego_id": int(ego_id),
            "tick": int(tick_index),
            "timestamp": float(timestamp),
            "gps": gps_list,
            "local_position": pos_list,
            "speed": float(speed),
            "compass": float(compass),
            "next_waypoint": waypoint,
            "next_waypoint_world": self._world_dict(near_world),
            "command": cmd_name,
            "distance_to_waypoint": dist,
        }
        if extra_fields:
            record.update(extra_fields)
        route_centerline_distance = record.get("nearest_route_centerline_distance")
        if route_centerline_distance is not None:
            route_centerline_distance = float(route_centerline_distance)
        self.tick_records.append(record)
        self.actual_positions.append(pos_list)
        self.distances.append(dist)
        if route_centerline_distance is not None:
            self.centerline_distances.append(route_centerline_distance)
        self.command_histogram[cmd_name] = self.command_histogram.get(cmd_name, 0) + 1

        divergence_distance = route_centerline_distance if route_centerline_distance is not None else dist
        if divergence_distance > self.divergence_threshold:
            if not self._last_over_threshold:
                self.divergence_events.append(
                    {
                        "tick": int(tick_index),
                        "timestamp": float(timestamp),
                        "distance": float(divergence_distance),
                        "distance_metric": "route_centerline" if route_centerline_distance is not None else "next_waypoint",
                        "command": cmd_name,
                        "position": pos_list,
                        "target_waypoint": waypoint,
                    }
                )
                self._last_over_threshold = True
        else:
            self._last_over_threshold = False

    def finalize(self):
        if self._finalized:
            return
        if self.tick_records:
            (self.base_dir / "runtime_ticks.json").write_text(json.dumps(self.tick_records, indent=2))
        summary = {
            "total_ticks": len(self.tick_records),
            "max_distance_to_next_waypoint": max(self.distances, default=0.0),
            "avg_distance_to_next_waypoint": float(np.mean(self.distances)) if self.distances else 0.0,
            "max_distance_to_route_centerline": max(self.centerline_distances, default=0.0),
            "avg_distance_to_route_centerline": float(np.mean(self.centerline_distances)) if self.centerline_distances else 0.0,
            "divergence_threshold": self.divergence_threshold,
            "divergence_events": self.divergence_events,
            "command_histogram": self.command_histogram,
            "plan_points": len(self.plan_entries),
        }
        (self.base_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._plot_paths()
        self._plot_distance_curve()
        self._finalized = True

    def _plot_paths(self):
        if not (self.plan_local_points or self.actual_positions):
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        plt.figure(figsize=(8, 8))
        if self.plan_local_points:
            plan_xy = np.array([pt["local_xy"] for pt in self.plan_local_points])
            plt.plot(plan_xy[:, 0], plan_xy[:, 1], "k--", label="Planner local path")
        if self.actual_positions:
            actual_xy = np.array(self.actual_positions)
            plt.plot(actual_xy[:, 0], actual_xy[:, 1], "r-", label="Actual trajectory")
            plt.scatter(actual_xy[0, 0], actual_xy[0, 1], c="green", marker="o", label="Start", zorder=5)
            plt.scatter(actual_xy[-1, 0], actual_xy[-1, 1], c="blue", marker="x", label="End", zorder=5)
        for event in self.divergence_events:
            plt.scatter(event["position"][0], event["position"][1], c="orange", marker="s", s=30, label="Divergence")
        plt.title("Planner vs. actual trajectory")
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(self.base_dir / "trajectory_overlay.png")
        plt.close()

    def _plot_distance_curve(self):
        if not self.tick_records:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        ticks = [rec["tick"] for rec in self.tick_records]
        plt.figure(figsize=(10, 4))
        plt.plot(ticks, self.distances, label="Distance to next waypoint")
        if self.centerline_distances:
            plt.plot(ticks[: len(self.centerline_distances)], self.centerline_distances, label="Distance to route centerline")
        plt.axhline(self.divergence_threshold, color="red", linestyle="--", label="Divergence threshold")
        plt.xlabel("Tick")
        plt.ylabel("Distance (m)")
        plt.title("Waypoint tracking error over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.base_dir / "distance_over_time.png")
        plt.close()

    @staticmethod
    def _command_name(command):
        if command is None:
            return "UNKNOWN"
        return getattr(command, "name", str(command))

    @staticmethod
    def _to_list(value):
        if value is None:
            return [None, None]
        if hasattr(value, "tolist"):
            return [float(x) for x in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        return [float(value), 0.0]

    @staticmethod
    def _unpack_plan_entry(entry):
        gps_info, command = None, None
        if isinstance(entry, tuple):
            first = entry[0]
            if isinstance(first, tuple):
                gps_info, command = first
            else:
                gps_info = first
                command = entry[1] if len(entry) > 1 else None
        else:
            gps_info = entry
        gps_dict = {}
        if isinstance(gps_info, dict):
            gps_dict = {k: float(v) for k, v in gps_info.items() if isinstance(v, (int, float))}
        elif hasattr(gps_info, "location"):
            gps_dict = {
                "x": float(gps_info.location.x),
                "y": float(gps_info.location.y),
                "z": float(gps_info.location.z),
            }
        return gps_dict, command

    @staticmethod
    def _world_dict(transform):
        if transform is None:
            return None
        return {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "yaw": float(transform.rotation.yaw),
        }
