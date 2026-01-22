"""
Pipeline Capabilities Definition - GROUND TRUTH

This module defines what the scenario generation pipeline CAN and CANNOT express.

"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum


# =============================================================================
# ACTUAL TOPOLOGICAL FEATURES (from step_01_crop/models.py CropFeatures)
# =============================================================================
# These are the ONLY map features the pipeline can detect and match:

@dataclass
class MapFeatures:
    """Features actually detected by compute_crop_features() in features.py"""
    # Turns detected in the crop (from path signatures)
    turns: List[str]  # ["straight", "left", "right", "uturn"]
    
    # Cardinal directions of path entries/exits
    entry_dirs: List[str]  # ["N", "S", "E", "W"]
    exit_dirs: List[str]
    
    # Junction type booleans
    has_oncoming_pair: bool      # Two straight paths with opposite entry directions
    is_t_junction: bool          # len(dirs) == 3
    is_four_way: bool            # len(dirs) >= 4
    has_merge_onto_same_road: bool  # Multiple entry turns lead to same exit road
    has_on_ramp: bool            # has_merge AND NOT is_four_way (heuristic)
    
    # Lane features
    lane_count_est: int          # 1-3 typically
    has_multi_lane: bool         # lane_count_est >= 2


# =============================================================================
# ACTUAL GEOMETRY SPEC (from step_01_crop/models.py GeometrySpec)
# =============================================================================
# This is what the LLM extractor produces from a description:

class TopologyType(Enum):
    """ONLY these topology types exist in the pipeline."""
    INTERSECTION = "intersection"  # General intersection (3+ way)
    T_JUNCTION = "t_junction"      # Exactly 3 directions
    CORRIDOR = "corridor"          # Straight road segment
    TWO_LANE_CORRIDOR = "two_lane_corridor"  # Bidirectional two-lane corridor
    HIGHWAY = "highway"            # Multi-lane high-speed road (3+ lanes)
    UNKNOWN = "unknown"            # Fallback


@dataclass
class GeometrySpecActual:
    """The ACTUAL geometry spec from llm_extractor.py"""
    topology: TopologyType
    degree: int  # 0, 3, or 4 (0 = unknown)
    
    # Maneuver requirements (counts)
    required_maneuvers: Dict[str, int]  # {"straight": 0-3, "left": 0-3, "right": 0-3}
    
    # Feature requirements
    needs_oncoming: bool
    needs_merge_onto_same_road: bool
    needs_on_ramp: bool
    needs_multi_lane: bool
    min_lane_count: int  # 1-3
    
    # Path length requirements (meters)
    min_entry_runup_m: float   # Default 28.0
    min_exit_runout_m: float   # Default 18.0
    
    # Preferences
    preferred_entry_cardinals: List[str]  # ["N", "S", "E", "W"] or []
    avoid_extra_intersections: bool


# =============================================================================
# ACTUAL NON-EGO ACTOR CAPABILITIES (from step_05_object_placer/prompts.py)
# =============================================================================

class ActorKind(Enum):
    """Actor types the pipeline can spawn."""
    STATIC_PROP = "static_prop"        # cones, barriers, debris, boxes
    PARKED_VEHICLE = "parked_vehicle"  # stationary vehicles blocking lanes
    WALKER = "walker"                  # pedestrians
    CYCLIST = "cyclist"                # bicycles
    NPC_VEHICLE = "npc_vehicle"        # Simple NPC (NOT an ego with its own path)


class MotionType(Enum):
    """Motion types from guardrails.py validation."""
    STATIC = "static"                  # No movement
    CROSS_PERPENDICULAR = "cross_perpendicular"  # Crosses lane perpendicular
    FOLLOW_LANE = "follow_lane"        # Moves along lane direction
    STRAIGHT_LINE = "straight_line"    # Moves between two anchors


class SpeedHint(Enum):
    """Speed hints - qualitative only, NOT precise."""
    STOPPED = "stopped"
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    ERRATIC = "erratic"
    UNKNOWN = "unknown"


class TimingPhase(Enum):
    """When an actor appears relative to ego path."""
    ON_APPROACH = "on_approach"
    AFTER_TURN = "after_turn"
    IN_INTERSECTION = "in_intersection"
    AFTER_EXIT = "after_exit"
    AFTER_MERGE = "after_merge"
    UNKNOWN = "unknown"


class LateralPosition(Enum):
    """Lateral positions from constants.py LATERAL_RELATIONS."""
    CENTER = "center"
    HALF_RIGHT = "half_right"
    RIGHT_EDGE = "right_edge"
    OFFROAD_RIGHT = "offroad_right"
    SIDEWALK_RIGHT = "sidewalk_right"  # pedestrian on right sidewalk
    HALF_LEFT = "half_left"
    LEFT_EDGE = "left_edge"
    OFFROAD_LEFT = "offroad_left"
    SIDEWALK_LEFT = "sidewalk_left"    # pedestrian on left sidewalk


class GroupPattern(Enum):
    """How multiple objects are arranged."""
    ACROSS_LANE = "across_lane"
    ALONG_LANE = "along_lane"
    DIAGONAL = "diagonal"
    UNKNOWN = "unknown"


class CrossingDirection(Enum):
    """For crossing motion only."""
    LEFT = "left"    # Crosses from right to left
    RIGHT = "right"  # Crosses from left to right


# =============================================================================
# ACTUAL EGO VEHICLE CONSTRAINTS (from step_03_path_picker/constraints.py)
# =============================================================================

class ConstraintType(Enum):
    """Inter-vehicle constraints that the path picker understands."""
    SAME_APPROACH_AS = "same_approach_as"
    OPPOSITE_APPROACH_OF = "opposite_approach_of"
    PERPENDICULAR_RIGHT_OF = "perpendicular_right_of"
    PERPENDICULAR_LEFT_OF = "perpendicular_left_of"
    SAME_EXIT_AS = "same_exit_as"
    SAME_ROAD_AS = "same_road_as"
    FOLLOW_ROUTE_OF = "follow_route_of"
    LEFT_LANE_OF = "left_lane_of"
    RIGHT_LANE_OF = "right_lane_of"
    MERGES_INTO_LANE_OF = "merges_into_lane_of"
    SAME_LANE_AS = "same_lane_as"


class EgoManeuver(Enum):
    """Maneuvers the path picker recognizes."""
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    LANE_CHANGE = "lane_change"
    UNKNOWN = "unknown"


# =============================================================================
# AVAILABLE CARLA TOWNS (from birdview_v2_cache directory)
# =============================================================================

AVAILABLE_TOWNS = ["Town01", "Town02", "Town05", "Town06", "Town07"]


# =============================================================================
# GROUND TRUTH PIPELINE CAPABILITIES
# =============================================================================

@dataclass
class PipelineCapabilities:
    """
    Complete specification of what the pipeline CAN and CANNOT express.
    This is the GROUND TRUTH derived from actual code analysis.
    """
    
    # === EGO VEHICLES ===
    # No hard limit on ego vehicles - the pipeline supports any reasonable number
    # Practical considerations: more vehicles = more complex path negotiation
    min_ego_vehicles: int = 1
    
    # === NON-EGO ACTORS ===
    # No hard limit - practical considerations for spawn point availability
    actor_kinds: Set[ActorKind] = field(default_factory=lambda: set(ActorKind))
    motion_types: Set[MotionType] = field(default_factory=lambda: set(MotionType))
    
    # === TOPOLOGY MATCHING ===
    # The pipeline matches descriptions to existing map regions
    # It can detect: intersection, t_junction, corridor, highway
    # It CANNOT create: roundabouts (no detection), signalized intersections
    supported_topologies: Set[TopologyType] = field(default_factory=lambda: {
        TopologyType.INTERSECTION,
        TopologyType.T_JUNCTION,
        TopologyType.CORRIDOR,
        TopologyType.HIGHWAY,
    })
    
    # Map feature detection
    can_detect_oncoming: bool = True
    can_detect_multi_lane: bool = True
    can_detect_on_ramp: bool = True  # Heuristic only
    can_detect_merge: bool = True
    can_detect_t_junction: bool = True
    can_detect_four_way: bool = True
    
    # === INTER-VEHICLE CONSTRAINTS ===
    constraint_types: Set[ConstraintType] = field(default_factory=lambda: set(ConstraintType))
    
    # === ACTOR PLACEMENT ===
    lateral_positions: Set[LateralPosition] = field(default_factory=lambda: set(LateralPosition))
    timing_phases: Set[TimingPhase] = field(default_factory=lambda: set(TimingPhase))
    s_along_range: Tuple[float, float] = (0.0, 1.0)  # Position along segment
    
    # === WHAT THE PIPELINE CANNOT DO ===
    hard_limitations: List[str] = field(default_factory=lambda: [
        # Map/Topology limitations
        "CANNOT create roundabouts - no roundabout detection in CropFeatures",
        "CANNOT create signalized intersections - no signal phase control",
        "CANNOT create highway diverge/off-ramps - only on-ramp heuristic exists",
        
        # Vehicle behavior limitations
        "CANNOT specify exact vehicle speeds in m/s or km/h - only qualitative hints",
        "CANNOT specify exact timing between vehicles in seconds",
        "CANNOT control NPC behavioral personality (aggressive, hesitant)",
        "CANNOT specify exact headway distances between vehicles",
        "CANNOT create complex multi-stage scenarios (beyond a single trigger -> action)",
        
        # Spatial limitations
        "CANNOT reference specific coordinates - only segment-relative positions",
        "CANNOT specify exact distances - only relative positions (s_along 0-1)",
        
        # Actor limitations
        "NPC vehicles only support simple trigger actions (start motion, hard brake, single lane change)",
        "NPC vehicles do NOT get their own picked paths like ego vehicles",
    ])


# Singleton instance
PIPELINE_CAPABILITIES = PipelineCapabilities()


# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

@dataclass
class MapRequirements:
    topology: TopologyType
    needs_oncoming: bool = False
    needs_multi_lane: bool = False
    needs_on_ramp: bool = False
    needs_merge: bool = False


@dataclass
class VariationAxis:
    name: str
    options: List[str]
    why: str = ""


@dataclass
class CategoryDefinition:
    """
    Lean scenario category definition for LLM prompt + deterministic validation.
    """
    name: str
    summary: str
    intent: str
    map: MapRequirements
    must_include: List[str]
    avoid: List[str]
    vary: List[VariationAxis] = field(default_factory=list)


# Honest assessment of each category
CATEGORY_DEFINITIONS: Dict[str, CategoryDefinition] = {
    # Legacy notes/conflict_via kept below each block for reference.
    "Intersection Deadlock Resolution": CategoryDefinition(
        name="Intersection Deadlock Resolution",
        summary="Uncontrolled intersection with multiple approaches and ambiguous right-of-way.",
        intent="Force multi-vehicle negotiation at an uncontrolled intersection where paths cross without clear priority.",
        map=MapRequirements(topology=TopologyType.INTERSECTION),
        must_include=[
            "Vehicle 1 must be from (entry_road=main) and Vehicle 2 from (entry_road=side)",
            "Every vehicle must have at least one conflicting interaction with another vehicle (i.e., no vehicle may be behaviorally independent).",
            "No vehicle may complete its maneuver without yielding to, blocking, or negotiating with at least one other vehicle.",
        ],
        avoid=[
            "Non-ego props, pedestrians, or cyclists",
        ],
        vary=[
            VariationAxis("ego_count", ["3", "4", "5", "6"], "number of egos entering from different approaches"),
            VariationAxis("approach_distribution", ["balanced across approaches", "heavy on one approach"], "how vehicles are distributed across approaches"),
        ],
    ),

    "Unprotected Left Turn": CategoryDefinition(
        name="Unprotected Left Turn",
        summary="Left turn across oncoming traffic without protection.",
        intent="Test gap acceptance and oncoming priority; conflict at junction center and exit lane.",
        map=MapRequirements(topology=TopologyType.INTERSECTION, needs_oncoming=True),
        must_include=[
            "Vehicle 1 turns left across oncoming traffic. Other vehicles may turn left/right/straight",
            "One vehicle mandatorly must have the opposite entry road of Vehicle 1 and continue straight (oncoming)",
            "Paths intersect at intersection center",
        ],
        avoid=[
            "Prop in the lane of travel of any vehicle, including props in the side of their lane",
        ],
        vary=[
            VariationAxis("ego_count", ["3", "4", "5", "6"], "total egos in the intersection scenario"),
            VariationAxis("oncoming_depth", ["single oncoming", "follow_route_of chain of 3-4"], "how many oncoming vehicles challenge the left turn"),
            VariationAxis("left_turners", ["single", "2-3 queued same approach", "opposing left-turner"], "distribution of left-turning vehicles"),
            VariationAxis("opposing_conflict", ["none", "opposing turner (same_exit_as clash)"], "whether an opposing turner competes for the exit lane"),
            VariationAxis("pedestrian", ["none", "walker crossing exit leg from sidewalk_right", "walker crossing exit leg from sidewalk_left"], "pedestrian involvement on the exit leg"),
            VariationAxis("occlusion", ["none", "parked_vehicle limiting visibility"], "visibility constraint level for the turn. vehicle type options: box truck, van, bus, delivery truck. vehicle must not block any vehicle paths"),
        ],
    ), 

    "Highway On-Ramp Merge": CategoryDefinition(
        name="Highway On-Ramp Merge",
        summary="Mainline highway with side on-ramp merging into traffic.",
        intent="Exercise merge negotiation between ramp and mainline vehicles on multi-lane highway geometry.",
        map=MapRequirements(topology=TopologyType.HIGHWAY, needs_on_ramp=True, needs_merge=True),
        must_include=[
            "Vehicle 1 is an on ramp vehicle (entry_road=side) merging into mainline (entry_road=main)",
            "At least one mainline vehicle in the lane that Vehicle 1 is merging into",
        ],
        avoid=[
            "Non-ego props, pedestrians, or cyclists",
        ],
        vary=[
            VariationAxis("ego_count", ["3", "4", "5", "6+"], "total vehicles across ramp and mainline"),
            VariationAxis("ramp_queue", ["single merging", "multiple merging"], "how many vehicles are queued on the ramp"),
            VariationAxis("ramp_adjacent_lane_platoon", ["single vehicle", "multiple vehicles"], "how many vehicles are queued in the adjacent lane next to the ramp"),
        ],
    ), 

    "Interactive Lane Change": CategoryDefinition(
        name="Interactive Lane Change",
        summary="Highway/corridor weaving with adjacent-lane interactions.",
        intent="Stress lane-change negotiations in multi-lane traffic without props.",
        map=MapRequirements(topology=TopologyType.HIGHWAY, needs_multi_lane=True),
        must_include=[
            "Vehicles in adjacent lanes",
            "Active lane-change relations (merges_into_lane_of / left/right lane of) for ALL vehicles",
        ],
        avoid=[
            "Non-ego props or pedestrians",
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4", "5"], "vehicles participating in weaving"),
            VariationAxis("lane_distribution", ["many vehicles attempt to merge into same lane", "merges are relatively even between lanes"], "how vehicles are distributed across lanes"),
        ],
    ), 

    "Blocked Lane (Obstacle)": CategoryDefinition(
        name="Blocked Lane (Obstacle)",
        summary="Corridor with lane blocked by parked/stationary object.",
        intent="Force lane change or negotiation around a blocked lane segment.",
        map=MapRequirements(topology=TopologyType.CORRIDOR, needs_multi_lane=True),
        must_include=[
            "Parked/stationary actor fully blocking or partially blocking a lane that a vehicle is travelling in",
            "Vehicle in adjacent lane relative to blocked lane (left/right lane of). Some vehicles may also have a different entry road and turn onto the blocked lane before the blockage.",
        ],
        avoid=[
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4", "5"], "vehicles navigating around the blockage"),
            VariationAxis("blockage_count", ["1", "2","3"], "number of separate blockages"),
            VariationAxis("blockage_lateral", ["center", "half_right","half_left"], "lateral placement of blockage"),
            VariationAxis("blockage_s_along", ["clustered", "far"], "if there are multiple blockages, this represents how blockages are spaced relative to each other"),
        ],
    ),

    "Lane Drop / Alternating Merge": CategoryDefinition(
        name="Lane Drop / Alternating Merge",
        summary="Corridor lane drop forcing zipper/alternating merge.",
        intent="Exercise alternating merges at a taper, optionally narrowed by props.",
        map=MapRequirements(topology=TopologyType.CORRIDOR, needs_multi_lane=True, needs_merge=True),
        must_include=[
            "For a specific lane, all vehicles merge out of a that lane into an adjacent lane at the same point",
            "For the lane being dropped, there must be an obstacle directly in front of where the vehicles start merging from (either cones or parked vehicle)",
        ],
        avoid=[
            "Pedestrians or cyclists",
            "Obstacles anywhere besides in front of the merge point of the lane being dropped",
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4", "5", "6", "7", "8"], "vehicles participating in the zipper merge"),
            VariationAxis("density in dropped_lane", ["sparse", "dense"], "how many vehicles are in the lane being dropped"),
            VariationAxis("obstacle_type", ["cones", "parked_vehicle"], "type of obstacle causing the lane drop"),
        ],
    ), 

    "Major/Minor Unsignalized Entry": CategoryDefinition(
        name="Major/Minor Unsignalized Entry",
        summary="T-junction yield from side street into main road traffic.",
        intent="Test gap acceptance for side-street entry into busy main road, with possible occlusion/pedestrians.",
        map=MapRequirements(topology=TopologyType.T_JUNCTION),
        must_include=[
            "Side street vehicle entering main road",
            "Main road traffic present (follow_route_of or opposite/perpendicular)",
        ],
        avoid=[
            "Signals controlling entry",
            "Props blocking vehicle paths",
        ],
        vary=[
            VariationAxis("ego_count", ["3", "4", "5"], "total vehicles at the T-junction"),
            VariationAxis("main_road_queue", ["single", "follow_route_of chain 2-4"], "volume of main-road traffic"),
            VariationAxis("side_street_queue", ["single", "follow_route_of chain 2"], "volume of side-street entrants"),
            VariationAxis("side_maneuver", ["left turn", "right turn"], "maneuver performed by the side-street vehicle"),
            VariationAxis("parked_vehicle", ["none", "parked_vehicle on opposite side of road blocking view"], "occluding parked vehicle presence, which does not block any vehicle paths"),
            VariationAxis("pedestrian", ["none", "walker crossing exit path"], "whether a pedestrian crosses the exit path"),
        ],
    ), 

    "Construction Zone": CategoryDefinition(
        name="Construction Zone",
        summary="Corridor work zone with cones/props narrowing lanes.",
        intent="Create forced merges and constrained paths using static props.",
        map=MapRequirements(topology=TopologyType.CORRIDOR, needs_multi_lane=True),
        must_include=[
            "Define one area / lane to be the construction zone, and do not place any props outside this area",
            "Clusters of multiple types of construction related static props in work zone",
            "Construction props must be selected ONLY from the following assets: cones (constructioncone, trafficcone01, trafficcone02), barriers (streetbarrier, barrel, chainbarrier, chainbarrierend), warning sign (trafficwarning), and at most one construction/utility vehicle (truck or van)."
            
        ],
        avoid=[
            "Pedestrians or cyclists",
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4", "5", "6"], "vehicles traversing the work zone"),
            VariationAxis("cone_count", ["3", "5", "7"], "number of cones placed"),
            VariationAxis("cone_pattern", ["along_lane", "diagonal"], "layout of the cone line"),
            VariationAxis("cone_lateral", ["center","left edge","right edge"], "lateral placement of cones"),
            VariationAxis("work_vehicle", ["none", "parked_vehicle"], "presence of a construction vehicle prop near the cones"),
        ],
    ),

    "Pedestrian Crosswalk": CategoryDefinition(
        name="Pedestrian Crosswalk",
        summary="Corridor crossing with pedestrian(s) interacting with vehicles.",
        intent="Test vehicle response to pedestrians crossing, with optional occlusion.",
        map=MapRequirements(topology=TopologyType.CORRIDOR),
        must_include=[
            "Pedestrian crossing perpendicular to vehicle path",
        ],
        avoid=[
            "Moving NPC vehicles or cyclists beyond egos",
            "Static props blocking vehicle path",
            "Static props not contributing to occlusion of pedestrian",
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4"], "vehicles approaching the crosswalk"),
            VariationAxis("walker_count", ["1", "2", "3"], "number of pedestrians crossing"),
            VariationAxis("walker_start", ["right sidewalk", "left sidewalk"], "which side the pedestrian starts from"),
            VariationAxis("occlusion", ["none", "parked_vehicle blocking view"], "whether an occluder exists blocking the view of a crossing pedestrian. this occluder must not block the paths of any vehicles"),
            VariationAxis("occlusion_type", ["box truck", "van", "bus", "delivery truck"], "type of occluding vehicle"),
        ],
    ), 

    "Overtaking on Two-Lane Road": CategoryDefinition(
        name="Overtaking on Two-Lane Road",
        summary="Corridor overtaking/pass maneuvers with adjacent/oncoming/obstacle factors.",
        intent="Exercise overtaking decisions with adjacent lane use, potential oncoming traffic, and obstacles.",
        map=MapRequirements(topology=TopologyType.TWO_LANE_CORRIDOR, needs_multi_lane=False),
        must_include=[
            "Prop blocking one lane of one vehicles path",
            "Another vehicle coming from opposite direction (oncoming)",
        ],
        avoid=[
            "Props on either side of the road that do not contribute to the overtaking scenario",
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4", "5"], "vehicles involved in the overtake scenario"),
            VariationAxis("obstacle", ["none", "parked_vehicle blocking pass lane"], "blocking obstacle forcing the pass"),
            VariationAxis("pedestrian", ["none", "walker crossing from sidewalk left or sidewalk right"], "pedestrian involvement during pass"),
            VariationAxis("cyclist", ["none", "cyclist riding in the lane opposite of the one that has the obstacle"], "cyclist involvement during pass"),
        ],
    ),
}


def get_available_categories() -> List[str]:
    """Return all supported categories."""
    return list(CATEGORY_DEFINITIONS.keys())
