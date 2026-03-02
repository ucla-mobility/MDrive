"""
Scenario Constraints - Structured IR Aligned with Pipeline

This module provides structured representations that map DIRECTLY to what
the pipeline can parse and execute. No aspirational features.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re

from .capabilities import (
    TopologyType, ActorKind, MotionType, SpeedHint, TimingPhase,
    LateralPosition, GroupPattern, ConstraintType, EgoManeuver,
    CATEGORY_DEFINITIONS, get_available_categories,
)


# =============================================================================
# EGO VEHICLE SPECIFICATION
# =============================================================================
# Maps directly to what step_03_path_picker/constraints.py can parse

@dataclass
class EgoVehicleSpec:
    """
    Specification for an ego vehicle.
    
    These map directly to the 'vehicles' array in the path picker constraints.
    """
    vehicle_id: str  # "Vehicle 1", "Vehicle 2", etc.
    maneuver: EgoManeuver = EgoManeuver.STRAIGHT
    lane_change_phase: str = "unknown"  # "before_intersection", "after_intersection", "unknown"
    entry_road: str = "unknown"  # "main", "side", "unknown"
    exit_road: str = "unknown"   # "main", "side", "unknown"


@dataclass
class InterVehicleConstraint:
    """
    Constraint between two ego vehicles.
    
    Maps directly to the 'constraints' array in the path picker.
    """
    constraint_type: ConstraintType
    vehicle_a: str  # "Vehicle 1"
    vehicle_b: str  # "Vehicle 2"
    
    def to_natural_language(self) -> str:
        """Convert to natural language for scenario description."""
        mapping = {
            ConstraintType.SAME_APPROACH_AS: f"{self.vehicle_a} approaches from the same direction as {self.vehicle_b}",
            ConstraintType.OPPOSITE_APPROACH_OF: f"{self.vehicle_a} approaches from the opposite direction of {self.vehicle_b}",
            ConstraintType.PERPENDICULAR_RIGHT_OF: f"{self.vehicle_a} approaches from the perpendicular road to the right of {self.vehicle_b}'s approach",
            ConstraintType.PERPENDICULAR_LEFT_OF: f"{self.vehicle_a} approaches from the perpendicular road to the left of {self.vehicle_b}'s approach",
            ConstraintType.SAME_EXIT_AS: f"{self.vehicle_a} exits in the same direction as {self.vehicle_b}",
            ConstraintType.SAME_ROAD_AS: f"{self.vehicle_a} ends up on the same road as {self.vehicle_b}",
            ConstraintType.FOLLOW_ROUTE_OF: f"{self.vehicle_a} follows behind {self.vehicle_b} in the same lane",
            ConstraintType.LEFT_LANE_OF: f"{self.vehicle_a} is in the lane to the left of {self.vehicle_b}",
            ConstraintType.RIGHT_LANE_OF: f"{self.vehicle_a} is in the lane to the right of {self.vehicle_b}",
            ConstraintType.MERGES_INTO_LANE_OF: f"{self.vehicle_a} changes lanes into {self.vehicle_b}'s lane",
        }
        return mapping.get(self.constraint_type, f"{self.constraint_type.value}({self.vehicle_a}, {self.vehicle_b})")


# =============================================================================
# NON-EGO ACTOR SPECIFICATION  
# =============================================================================
# Maps directly to what step_05_object_placer/prompts.py Stage 1 can parse

@dataclass
class NonEgoActorSpec:
    """
    Specification for a non-ego actor (obstacle, pedestrian, etc.).
    
    Maps directly to the entity schema in object placer Stage 1.
    """
    # Identity
    actor_id: str  # e.g., "parked truck", "pedestrian", "traffic cones"
    kind: ActorKind
    quantity: int = 1
    
    # Spatial placement (relative to an ego vehicle's path)
    affects_vehicle: Optional[str] = None  # "Vehicle 1", etc.
    timing_phase: TimingPhase = TimingPhase.UNKNOWN
    lateral_position: LateralPosition = LateralPosition.CENTER
    
    # For groups (quantity > 1)
    group_pattern: GroupPattern = GroupPattern.UNKNOWN
    start_lateral: Optional[LateralPosition] = None
    end_lateral: Optional[LateralPosition] = None
    
    # Motion
    motion: MotionType = MotionType.STATIC
    speed: SpeedHint = SpeedHint.UNKNOWN
    crossing_direction: Optional[str] = None  # "left" or "right" for crossing motion
    
    # For NPC vehicles only
    direction_relative_to: Optional[str] = None  # "same" or "opposite"
    
    def to_natural_language(self) -> str:
        """Convert to natural language fragment for scenario description."""
        parts = []

        def _lateral_phrase(pos: Optional[LateralPosition]) -> str:
            if pos is None:
                return ""
            mapping = {
                LateralPosition.CENTER: "center",
                LateralPosition.HALF_RIGHT: "half right",
                LateralPosition.RIGHT_EDGE: "right edge",
                LateralPosition.OFFROAD_RIGHT: "off the right side",
                LateralPosition.HALF_LEFT: "half left",
                LateralPosition.LEFT_EDGE: "left edge",
                LateralPosition.OFFROAD_LEFT: "off the left side",
            }
            return mapping.get(pos, "")
        
        # Quantity and type
        if self.quantity > 1:
            parts.append(f"{self.quantity} {self.actor_id}")
        else:
            parts.append(f"a {self.actor_id}")
        
        # Position relative to vehicle
        if self.affects_vehicle:
            if self.timing_phase == TimingPhase.ON_APPROACH:
                parts.append(f"on {self.affects_vehicle}'s approach")
            elif self.timing_phase == TimingPhase.AFTER_TURN:
                parts.append(f"after {self.affects_vehicle}'s turn")
            elif self.timing_phase == TimingPhase.IN_INTERSECTION:
                parts.append("in the intersection")
            elif self.timing_phase == TimingPhase.AFTER_EXIT:
                parts.append(f"on {self.affects_vehicle}'s exit road")
            elif self.timing_phase == TimingPhase.AFTER_MERGE:
                parts.append(f"after the merge region on {self.affects_vehicle}'s exit road")
        else:
            if self.timing_phase == TimingPhase.IN_INTERSECTION:
                parts.append("in the intersection")
            elif self.timing_phase == TimingPhase.AFTER_EXIT:
                parts.append("on the exit road")
            elif self.timing_phase == TimingPhase.AFTER_MERGE:
                parts.append("after the merge region")
        
        # Lateral position
        if self.lateral_position != LateralPosition.CENTER:
            pos_map = {
                LateralPosition.HALF_RIGHT: "positioned to the right of center",
                LateralPosition.RIGHT_EDGE: "at the right edge of the lane",
                LateralPosition.OFFROAD_RIGHT: "off the right side of the road",
                LateralPosition.HALF_LEFT: "positioned to the left of center", 
                LateralPosition.LEFT_EDGE: "at the left edge of the lane",
                LateralPosition.OFFROAD_LEFT: "off the left side of the road",
            }
            parts.append(pos_map.get(self.lateral_position, ""))

        # Group pattern
        if self.quantity > 1 and self.group_pattern != GroupPattern.UNKNOWN:
            if self.group_pattern == GroupPattern.ACROSS_LANE:
                parts.append("arranged across the lane width")
            elif self.group_pattern == GroupPattern.ALONG_LANE:
                parts.append("arranged along the lane")
            elif self.group_pattern == GroupPattern.DIAGONAL:
                start_phrase = _lateral_phrase(self.start_lateral)
                end_phrase = _lateral_phrase(self.end_lateral)
                if start_phrase and end_phrase:
                    parts.append(f"arranged diagonally from the {start_phrase} to the {end_phrase}")
                else:
                    parts.append("arranged diagonally across the lane")

        # Motion
        if self.motion == MotionType.CROSS_PERPENDICULAR:
            dir_str = ""
            if self.crossing_direction == "left":
                dir_str = "from right to left"
            elif self.crossing_direction == "right":
                dir_str = "from left to right"
            parts.append(f"crossing the road {dir_str}".strip())
        elif self.motion == MotionType.FOLLOW_LANE:
            parts.append("moving along the lane")
        elif self.motion == MotionType.STRAIGHT_LINE:
            parts.append("moving along a straight line")

        # Direction relative to ego (NPC vehicles only)
        if self.direction_relative_to and self.affects_vehicle:
            if self.direction_relative_to == "same":
                parts.append(f"traveling in the same direction as {self.affects_vehicle}")
            elif self.direction_relative_to == "opposite":
                parts.append(f"oncoming relative to {self.affects_vehicle}")
        
        return " ".join(filter(None, parts))


# =============================================================================
# COMPLETE SCENARIO SPECIFICATION
# =============================================================================

@dataclass
class ScenarioSpec:
    """
    Complete specification for a scenario.
    
    This is the structured IR that:
    1. Can be validated against pipeline capabilities
    2. Can be converted to natural language for the pipeline
    3. Can be generated creatively within bounds
    """
    # Metadata
    category: str
    
    # Map requirements
    topology: TopologyType
    needs_oncoming: bool = False
    needs_multi_lane: bool = False
    needs_on_ramp: bool = False
    needs_merge: bool = False
    
    # Ego vehicles and their relationships
    ego_vehicles: List[EgoVehicleSpec] = field(default_factory=list)
    vehicle_constraints: List[InterVehicleConstraint] = field(default_factory=list)
    
    # Non-ego actors
    actors: List[NonEgoActorSpec] = field(default_factory=list)
    
    # Natural language output (generated from above)
    description: str = ""
    
    def generate_description(self) -> str:
        """
        Generate a natural language description from the structured spec.
        This produces text that the pipeline can parse.
        
        Uses intelligent grouping to clearly describe which vehicles share approach roads.
        """
        parts = []

        # Build approach road groups from constraints
        approach_groups = self._compute_approach_groups()
        described_vehicles = set()
        
        # Helper to describe a vehicle's maneuver
        def maneuver_phrase(veh: EgoVehicleSpec) -> str:
            if veh.maneuver == EgoManeuver.LEFT:
                if self.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                    return "turns left at the intersection"
                return "turns left"
            elif veh.maneuver == EgoManeuver.RIGHT:
                if self.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                    return "turns right at the intersection"
                return "turns right"
            elif veh.maneuver == EgoManeuver.LANE_CHANGE:
                if self.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                    if veh.lane_change_phase == "before_intersection":
                        return "changes lanes before the intersection"
                    elif veh.lane_change_phase == "after_intersection":
                        return "changes lanes after the intersection"
                return "changes lanes"
            else:
                if self.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                    return "goes straight through the intersection"
                return "continues straight"
        
        # Helper to describe approach context
        def cardinal_phrase(veh: EgoVehicleSpec) -> str:
            """
            Provide a canonical cardinal phrasing for entry/exit to reduce ambiguity
            for downstream path picking. We map main vs side to perpendicular axes.
            """
            if self.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                if veh.entry_road == "main":
                    entry = "from west to east (main road)"
                elif veh.entry_road == "side":
                    entry = "from south to north (side road)"
                else:
                    entry = "from an unspecified approach"

                if veh.exit_road == "main":
                    exit_dir = "continuing eastbound"
                elif veh.exit_road == "side":
                    exit_dir = "continuing northbound"
                else:
                    exit_dir = "continuing through"

                return f"enters {entry} and {exit_dir}"

            if self.topology == TopologyType.HIGHWAY:
                if veh.entry_road == "main":
                    return "travels along the mainline"
                if veh.entry_road == "side":
                    return "enters from an on-ramp onto the mainline"
            return "approaches"

        def approach_phrase(veh: EgoVehicleSpec) -> str:
            if veh.entry_road == "main":
                if self.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                    return "approaches on the main road"
                return "travels along the main road"
            elif veh.entry_road == "side":
                if self.needs_on_ramp or self.topology == TopologyType.HIGHWAY:
                    return "travels from the on-ramp"
                if self.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
                    return "approaches from the side road"
                return "travels from the side road"
            else:
                if self.topology == TopologyType.T_JUNCTION:
                    return "approaches the T-junction"
                elif self.topology == TopologyType.INTERSECTION:
                    return "approaches the intersection"
                return "travels"

        def vehicle_list_phrase(vehicle_ids: List[str]) -> str:
            if not vehicle_ids:
                return "approaching vehicles"
            if len(vehicle_ids) == 1:
                return vehicle_ids[0]
            if len(vehicle_ids) == 2:
                return f"{vehicle_ids[0]} and {vehicle_ids[1]}"
            return ", ".join(vehicle_ids[:-1]) + f", and {vehicle_ids[-1]}"
        
        # First, describe vehicles in approach groups (vehicles sharing same approach road)
        for group in approach_groups:
            if len(group) >= 2:
                # Multiple vehicles share this approach road - describe them together
                group_veh_ids = sorted(group)
                group_vehicles = [v for v in self.ego_vehicles if v.vehicle_id in group_veh_ids]
                
                if not group_vehicles:
                    continue
                
                # Use the first vehicle's entry_road for the group approach description
                first_veh = group_vehicles[0]
                
                # Build the grouped approach description with proper grammar
                if len(group_vehicles) == 2:
                    veh_list = f"{group_vehicles[0].vehicle_id} and {group_vehicles[1].vehicle_id}"
                else:
                    veh_list = ", ".join(v.vehicle_id for v in group_vehicles[:-1])
                    veh_list += f", and {group_vehicles[-1].vehicle_id}"
                
                # Describe the shared approach
                approach_verb = "both approach" if len(group_vehicles) == 2 else "all approach"
                if first_veh.entry_road == "main":
                    parts.append(f"{veh_list} {approach_verb} from the same road (the main road), entering west to east.")
                elif first_veh.entry_road == "side":
                    if self.needs_on_ramp or self.topology == TopologyType.HIGHWAY:
                        parts.append(f"{veh_list} {approach_verb} from the same road (the on-ramp).")
                    else:
                        parts.append(f"{veh_list} {approach_verb} from the same road (the side road), entering south to north.")
                else:
                    parts.append(f"{veh_list} {approach_verb} from the same road.")
                
                # Now describe each vehicle's individual maneuver
                for veh in group_vehicles:
                    parts.append(f"{veh.vehicle_id} {maneuver_phrase(veh)} ({cardinal_phrase(veh)}).")
                    described_vehicles.add(veh.vehicle_id)
        
        # Describe remaining vehicles that are not in any same_approach group
        # Check if they have opposite_approach_of constraints to describe them with more context
        opposite_relations = self._get_opposite_approach_relations()
        perpendicular_relations = self._get_perpendicular_approach_relations()
        
        # Track which vehicles have been mentioned as "reference" in opposite/perpendicular relations
        # to avoid saying "A is opposite B" and then "B is opposite A"
        used_as_reference = set()
        
        for veh in self.ego_vehicles:
            if veh.vehicle_id not in described_vehicles:
                veh_parts = [f"{veh.vehicle_id}"]
                
                # Check if this vehicle has opposite approach relations
                opposite_to = opposite_relations.get(veh.vehicle_id, [])
                perpendicular_to = perpendicular_relations.get(veh.vehicle_id, [])
                
                # Filter out vehicles we've already used as reference
                opposite_to = [v for v in opposite_to if v not in used_as_reference]
                perpendicular_to = [v for v in perpendicular_to if v not in used_as_reference]
                
                if opposite_to:
                    # Describe as approaching from opposite direction
                    # Only reference vehicles that have been described OR are in approach groups
                    ref_vehicle = None
                    for ref in opposite_to:
                        if ref in described_vehicles:
                            ref_vehicle = ref
                            break
                    
                    if ref_vehicle:
                        veh_parts.append(f"approaches from the opposite direction of {ref_vehicle}")
                        used_as_reference.add(veh.vehicle_id)
                    else:
                        veh_parts.append(approach_phrase(veh))
                elif perpendicular_to:
                    # Describe as approaching from perpendicular road
                    veh_parts.append("approaches from a perpendicular road")
                    used_as_reference.add(veh.vehicle_id)
                else:
                    veh_parts.append(approach_phrase(veh))
                
                veh_parts.append(f"and {maneuver_phrase(veh)} ({cardinal_phrase(veh)})")
                parts.append(" ".join(veh_parts) + ".")
                described_vehicles.add(veh.vehicle_id)
        
        # Add exit relationship constraint for vehicles that need it
        # (same_exit_as is still useful to state explicitly)
        exit_constraints_described = set()
        for constraint in self.vehicle_constraints:
            if constraint.constraint_type == ConstraintType.SAME_EXIT_AS:
                parts.append(constraint.to_natural_language() + ".")
                exit_constraints_described.add((constraint.vehicle_a, constraint.vehicle_b))
        
        # Describe remaining constraints (excluding those already covered via grouping/vehicle descriptions)
        for constraint in self.vehicle_constraints:
            # Skip same_approach_as - we've handled these via grouping
            if constraint.constraint_type == ConstraintType.SAME_APPROACH_AS:
                continue
            # Skip follow_route_of - handled via same_approach grouping
            if constraint.constraint_type == ConstraintType.FOLLOW_ROUTE_OF:
                continue
            # Skip exit constraints we already described
            if constraint.constraint_type == ConstraintType.SAME_EXIT_AS:
                continue
            # Skip opposite_approach_of if we described it in the vehicle section
            if constraint.constraint_type == ConstraintType.OPPOSITE_APPROACH_OF:
                # We describe this when the vehicle is NOT in an approach group
                # Only add it explicitly if not already covered
                continue
            # Skip perpendicular constraints - covered in vehicle description
            if constraint.constraint_type in {ConstraintType.PERPENDICULAR_LEFT_OF, ConstraintType.PERPENDICULAR_RIGHT_OF}:
                continue
            parts.append(constraint.to_natural_language() + ".")
        
        # Describe non-ego actors
        for actor in self.actors:
            parts.append(actor.to_natural_language() + ".")

        self.description = " ".join(parts)
        return self.description
    
    def _compute_approach_groups(self) -> List[set]:
        """
        Compute groups of vehicles that share the same approach road.
        Groups are based on matching entry_road values (main/side), augmented
        by explicit same_approach_as constraints.
        
        Returns a list of sets, where each set contains vehicle IDs that
        share the same approach road.
        """
        # Build vehicle lookup
        vehicle_by_id = {v.vehicle_id: v for v in self.ego_vehicles}
        vehicle_ids = set(vehicle_by_id.keys())
        
        # Initialize parent for union-find
        parent = {vid: vid for vid in vehicle_ids}
        
        def find(x: str) -> str:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a: str, b: str):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb
        
        # First, group vehicles by their explicit entry_road values (main/side)
        # Vehicles with the same non-unknown entry_road should be grouped together
        entry_road_groups: Dict[str, List[str]] = {}
        for vid, veh in vehicle_by_id.items():
            er = veh.entry_road
            if er in ("main", "side"):
                if er not in entry_road_groups:
                    entry_road_groups[er] = []
                entry_road_groups[er].append(vid)
        
        # Union vehicles with the same entry_road
        for road_type, vids in entry_road_groups.items():
            if len(vids) >= 2:
                for vid in vids[1:]:
                    union(vids[0], vid)
        
        # Also honor explicit same_approach_as constraints (these are definitive)
        for c in self.vehicle_constraints:
            if c.constraint_type == ConstraintType.SAME_APPROACH_AS:
                if c.vehicle_a in vehicle_ids and c.vehicle_b in vehicle_ids:
                    union(c.vehicle_a, c.vehicle_b)
        
        # Build groups
        groups_dict: Dict[str, set] = {}
        for vid in vehicle_ids:
            root = find(vid)
            if root not in groups_dict:
                groups_dict[root] = set()
            groups_dict[root].add(vid)
        
        # Return only groups with 2+ vehicles (single-vehicle groups don't need special handling)
        return [g for g in groups_dict.values() if len(g) >= 2]
    
    def _get_opposite_approach_relations(self) -> Dict[str, List[str]]:
        """
        Get a mapping of vehicle_id -> list of vehicle_ids it approaches opposite to.
        """
        relations: Dict[str, List[str]] = {}
        for c in self.vehicle_constraints:
            if c.constraint_type == ConstraintType.OPPOSITE_APPROACH_OF:
                # Both vehicles are opposite to each other
                if c.vehicle_a not in relations:
                    relations[c.vehicle_a] = []
                if c.vehicle_b not in relations:
                    relations[c.vehicle_b] = []
                relations[c.vehicle_a].append(c.vehicle_b)
                relations[c.vehicle_b].append(c.vehicle_a)
        return relations
    
    def _get_perpendicular_approach_relations(self) -> Dict[str, List[str]]:
        """
        Get a mapping of vehicle_id -> list of vehicle_ids it approaches perpendicular to.
        """
        relations: Dict[str, List[str]] = {}
        for c in self.vehicle_constraints:
            if c.constraint_type in {ConstraintType.PERPENDICULAR_LEFT_OF, ConstraintType.PERPENDICULAR_RIGHT_OF}:
                # Both vehicles are perpendicular to each other
                if c.vehicle_a not in relations:
                    relations[c.vehicle_a] = []
                if c.vehicle_b not in relations:
                    relations[c.vehicle_b] = []
                relations[c.vehicle_a].append(c.vehicle_b)
                relations[c.vehicle_b].append(c.vehicle_a)
        return relations
    
    def _describe_conflict(self) -> str:
        """Generate conflict description based on scenario type."""
        if len(self.ego_vehicles) >= 2:
            if self.needs_on_ramp or self.needs_merge:
                return "The vehicles reach the merge region with overlapping timing, creating an ambiguous merge interaction."
            elif self.needs_oncoming:
                return "Their arrival windows overlap, creating ambiguous priority negotiation."
            elif self.needs_multi_lane:
                return "The lane positioning creates ambiguous negotiation for available gaps."
            else:
                return "Their paths interact, creating coordination requirements."
        return ""


# =============================================================================
# VALIDATION
# =============================================================================

def validate_spec(spec: ScenarioSpec) -> Tuple[bool, List[str]]:
    """
    Validate that a scenario spec is expressible by the pipeline.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    # Check category is supported
    if spec.category not in get_available_categories():
        errors.append(f"Category '{spec.category}' is not supported by current pipeline")
    
    # Check topology is supported
    if spec.topology not in {
        TopologyType.INTERSECTION,
        TopologyType.T_JUNCTION,
        TopologyType.CORRIDOR,
        TopologyType.TWO_LANE_CORRIDOR,
        TopologyType.HIGHWAY,
        TopologyType.ROUNDABOUT,
    }:
        errors.append(f"Topology '{spec.topology.value}' is not supported")
    
    # Check ego vehicle count
    if len(spec.ego_vehicles) < 1:
        errors.append("Must have at least 1 ego vehicle")
    # No upper limit on ego vehicles - the pipeline can handle any reasonable number
    
    # Check actor count
    if len(spec.actors) > 8:
        errors.append(f"Too many non-ego actors: {len(spec.actors)} > 8")
    
    # Validate ego vehicle specs
    for veh in spec.ego_vehicles:
        if not re.match(r"Vehicle \d+", veh.vehicle_id):
            errors.append(f"Invalid vehicle_id format: {veh.vehicle_id}")
    
    # Validate constraints reference valid vehicles
    valid_ids = {v.vehicle_id for v in spec.ego_vehicles}
    for c in spec.vehicle_constraints:
        if c.vehicle_a not in valid_ids:
            errors.append(f"Constraint references unknown vehicle: {c.vehicle_a}")
        if c.vehicle_b not in valid_ids:
            errors.append(f"Constraint references unknown vehicle: {c.vehicle_b}")
    
    # Validate actor specs
    for actor in spec.actors:
        if actor.motion == MotionType.CROSS_PERPENDICULAR and actor.kind not in {ActorKind.WALKER, ActorKind.CYCLIST}:
            errors.append(f"Cross perpendicular motion only valid for walkers/cyclists, not {actor.kind.value}")
    
    # Check that required features are consistent
    cat = CATEGORY_DEFINITIONS.get(spec.category)
    if cat:
        if cat.map.topology and spec.topology != cat.map.topology:
            errors.append(f"Category {spec.category} requires topology {cat.map.topology.value}")
        if cat.map.needs_oncoming and not spec.needs_oncoming:
            errors.append(f"Category {spec.category} requires oncoming traffic")
        if cat.map.needs_multi_lane and not spec.needs_multi_lane:
            errors.append(f"Category {spec.category} requires multi-lane")
        if cat.map.needs_on_ramp and not spec.needs_on_ramp:
            errors.append(f"Category {spec.category} requires on-ramp")
        if cat.map.needs_merge and not spec.needs_merge:
            errors.append(f"Category {spec.category} requires merge")
    
    return (len(errors) == 0, errors)


def spec_to_dict(spec: ScenarioSpec) -> Dict[str, Any]:
    """Convert spec to dictionary for JSON serialization."""
    return {
        "category": spec.category,
        "topology": spec.topology.value,
        "needs_oncoming": spec.needs_oncoming,
        "needs_multi_lane": spec.needs_multi_lane,
        "needs_on_ramp": spec.needs_on_ramp,
        "needs_merge": spec.needs_merge,
        "ego_vehicles": [
            {
                "vehicle_id": v.vehicle_id,
                "maneuver": v.maneuver.value,
                "lane_change_phase": v.lane_change_phase,
                "entry_road": v.entry_road,
                "exit_road": v.exit_road,
            }
            for v in spec.ego_vehicles
        ],
        "vehicle_constraints": [
            {
                "type": c.constraint_type.value,
                "a": c.vehicle_a,
                "b": c.vehicle_b,
            }
            for c in spec.vehicle_constraints
        ],
        "actors": [
            {
                "actor_id": a.actor_id,
                "kind": a.kind.value,
                "quantity": a.quantity,
                "group_pattern": a.group_pattern.value,
                "start_lateral": a.start_lateral.value if a.start_lateral else None,
                "end_lateral": a.end_lateral.value if a.end_lateral else None,
                "affects_vehicle": a.affects_vehicle,
                "timing_phase": a.timing_phase.value,
                "lateral_position": a.lateral_position.value,
                "motion": a.motion.value,
                "speed": a.speed.value,
                "crossing_direction": a.crossing_direction,
                "direction_relative_to": a.direction_relative_to,
            }
            for a in spec.actors
        ],
        "description": spec.description,
    }


def spec_from_dict(d: Dict[str, Any]) -> ScenarioSpec:
    """Parse spec from dictionary."""
    ego_vehicles = [
        EgoVehicleSpec(
            vehicle_id=v["vehicle_id"],
            maneuver=EgoManeuver(v.get("maneuver", "straight")),
            lane_change_phase=v.get("lane_change_phase", "unknown"),
            entry_road=v.get("entry_road", "unknown"),
            exit_road=v.get("exit_road", "unknown"),
        )
        for v in d.get("ego_vehicles", [])
    ]
    
    vehicle_constraints = [
        InterVehicleConstraint(
            constraint_type=ConstraintType(c["type"]),
            vehicle_a=c["a"],
            vehicle_b=c["b"],
        )
        for c in d.get("vehicle_constraints", [])
    ]
    
    actors = [
        NonEgoActorSpec(
            actor_id=a["actor_id"],
            kind=ActorKind(a["kind"]),
            quantity=a.get("quantity", 1),
            affects_vehicle=a.get("affects_vehicle"),
            timing_phase=TimingPhase(a.get("timing_phase", "unknown")),
            lateral_position=LateralPosition(a.get("lateral_position", "center")),
            group_pattern=GroupPattern(a.get("group_pattern", "unknown")),
            start_lateral=LateralPosition(a.get("start_lateral")) if a.get("start_lateral") else None,
            end_lateral=LateralPosition(a.get("end_lateral")) if a.get("end_lateral") else None,
            motion=MotionType(a.get("motion", "static")),
            speed=SpeedHint(a.get("speed", "unknown")),
            crossing_direction=a.get("crossing_direction"),
            direction_relative_to=a.get("direction_relative_to"),
        )
        for a in d.get("actors", [])
    ]
    
    return ScenarioSpec(
        category=d["category"],
        topology=TopologyType(d["topology"]),
        needs_oncoming=d.get("needs_oncoming", False),
        needs_multi_lane=d.get("needs_multi_lane", False),
        needs_on_ramp=d.get("needs_on_ramp", False),
        needs_merge=d.get("needs_merge", False),
        ego_vehicles=ego_vehicles,
        vehicle_constraints=vehicle_constraints,
        actors=actors,
        description=d.get("description", ""),
    )
