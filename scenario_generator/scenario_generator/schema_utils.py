"""
Schema utilities for JSON-based scenario specifications.

These helpers keep the schema aligned with the existing pipeline primitives.
"""

import sys
from pathlib import Path
from typing import Dict

# Ensure scenario_generator/ is on sys.path for pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step_01_crop.models import GeometrySpec

from .capabilities import TopologyType
from .constraints import ScenarioSpec, EgoManeuver


def geometry_spec_from_scenario_spec(spec: ScenarioSpec) -> GeometrySpec:
    """
    Build a GeometrySpec directly from a structured ScenarioSpec.
    This bypasses the LLM geometry extractor for crop selection.
    """
    required_maneuvers: Dict[str, int] = {"straight": 0, "left": 0, "right": 0}
    for veh in spec.ego_vehicles:
        if veh.maneuver == EgoManeuver.STRAIGHT:
            required_maneuvers["straight"] += 1
        elif veh.maneuver == EgoManeuver.LEFT:
            required_maneuvers["left"] += 1
        elif veh.maneuver == EgoManeuver.RIGHT:
            required_maneuvers["right"] += 1

    topology = spec.topology.value if isinstance(spec.topology, TopologyType) else str(spec.topology)
    if spec.topology == TopologyType.HIGHWAY:
        degree = 0  # Highways don't have junction degree
    elif spec.topology == TopologyType.T_JUNCTION:
        degree = 3
    elif spec.topology == TopologyType.INTERSECTION:
        degree = 4
    else:
        degree = 0

    needs_merge_onto_same_road = bool(spec.needs_merge or spec.needs_on_ramp)
    needs_multi_lane = bool(spec.needs_multi_lane)
    # Highways require minimum 3 lanes; two-lane corridors force 2; others default to 2 for multi-lane
    if spec.topology == TopologyType.HIGHWAY:
        min_lane_count = 3
    elif spec.topology == TopologyType.TWO_LANE_CORRIDOR:
        min_lane_count = 2
    else:
        min_lane_count = 2 if needs_multi_lane else 1

    return GeometrySpec(
        topology=topology,
        degree=degree,
        required_maneuvers=required_maneuvers,
        needs_oncoming=bool(spec.needs_oncoming),
        needs_merge_onto_same_road=needs_merge_onto_same_road,
        needs_on_ramp=bool(spec.needs_on_ramp),
        needs_multi_lane=needs_multi_lane,
        min_lane_count=min_lane_count,
        min_entry_runup_m=28.0,
        min_exit_runout_m=18.0,
        preferred_entry_cardinals=[],
        avoid_extra_intersections=True,
        confidence=1.0,
        notes="schema_spec",
    )


def description_from_spec(spec: ScenarioSpec) -> str:
    """Generate or return a cached description from a ScenarioSpec."""
    if spec.description:
        return spec.description
    return spec.generate_description()
