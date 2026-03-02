"""
Ultra-minimal agent for smoke-testing run_custom_eval startup.

Purpose:
- Keep dependencies as close to evaluator baseline as possible.
- Confirm routes/world/sensors can initialize and the loop can tick.

Behavior:
- Registers only a speedometer pseudo-sensor.
- Returns neutral controls for each ego vehicle.
"""

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return "MinimalAgent"


class MinimalAgent(AutonomousAgent):
    """Load-test agent with minimal runtime surface."""

    def setup(self, path_to_conf_file, ego_vehicles_num=1):
        # Keep to SENSORS track; avoids OpenDRIVE-map sensor requirements.
        self.track = Track.SENSORS
        # AgentWrapper expects this attribute.
        self.ego_vehicles_num = int(ego_vehicles_num)
        self.agent_name = "minimal"

    def sensors(self):
        # One lightweight pseudo-sensor; avoids camera/lidar/image dependencies.
        if self.ego_vehicles_num <= 0:
            return []
        return [
            {
                "type": "sensor.speedometer",
                "id": "speed",
                "reading_frequency": 20,
            }
        ]

    def run_step(self, input_data, timestamp):
        # One neutral control command per ego vehicle.
        controls = []
        for _ in range(self.ego_vehicles_num):
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False
            controls.append(control)
        return controls

    def destroy(self):
        return

