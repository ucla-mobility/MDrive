LATERAL_RELATIONS = [
    "center",
    "half_right",
    "right_edge",
    "offroad_right",
    "sidewalk_right",  # alias for offroad_right (pedestrian-friendly term)
    "right_lane",      # vehicle in adjacent right lane (one full lane width)
    "half_left",
    "left_edge",
    "offroad_left",
    "sidewalk_left",   # alias for offroad_left (pedestrian-friendly term)
    "left_lane",       # vehicle in adjacent left lane (one full lane width)
]

# Lane width heuristic (meters)
LANE_WIDTH_M = 3.5

# Sidewalk offset from road edge (meters)
SIDEWALK_OFFSET_M = 1.5 * LANE_WIDTH_M  # ~5.25m from lane center to sidewalk

# Map qualitative to meters (right positive using right_normal_world)
LATERAL_TO_M = {
    "center": 0.0,
    "half_right": +0.25 * LANE_WIDTH_M,
    "right_edge": +0.45 * LANE_WIDTH_M,
    "offroad_right": +1.10 * LANE_WIDTH_M,
    "sidewalk_right": +SIDEWALK_OFFSET_M,  # pedestrian on right sidewalk
    "right_lane": +1.0 * LANE_WIDTH_M,     # center of adjacent right lane
    "half_left": -0.25 * LANE_WIDTH_M,
    "left_edge": -0.45 * LANE_WIDTH_M,
    "offroad_left": -1.10 * LANE_WIDTH_M,
    "sidewalk_left": -SIDEWALK_OFFSET_M,   # pedestrian on left sidewalk
    "left_lane": -1.0 * LANE_WIDTH_M,      # center of adjacent left lane
}

# Aliases for natural language parsing
LATERAL_ALIASES = {
    "side_of_road": "sidewalk_right",
    "side of the road": "sidewalk_right",
    "roadside": "sidewalk_right",
    "curb": "sidewalk_right",
    "sidewalk": "sidewalk_right",
    "pavement": "sidewalk_right",
    "right_sidewalk": "sidewalk_right",
    "left_sidewalk": "sidewalk_left",
    "right sidewalk": "sidewalk_right",
    "left sidewalk": "sidewalk_left",
    # Adjacent lane aliases
    "lane to the left": "left_lane",
    "lane_to_the_left": "left_lane",
    "adjacent left lane": "left_lane",
    "adjacent_left_lane": "left_lane",
    "left adjacent lane": "left_lane",
    "lane to the right": "right_lane",
    "lane_to_the_right": "right_lane",
    "adjacent right lane": "right_lane",
    "adjacent_right_lane": "right_lane",
    "right adjacent lane": "right_lane",
}

__all__ = [
    "LANE_WIDTH_M",
    "LATERAL_RELATIONS",
    "LATERAL_TO_M",
    "LATERAL_ALIASES",
    "SIDEWALK_OFFSET_M",
]
