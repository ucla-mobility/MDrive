"""
Pipeline Capabilities Definition - GROUND TRUTH

This module defines what the scenario generation pipeline CAN and CANNOT express.

"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Literal
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
    ROUNDABOUT = "roundabout"      # Circular intersection (only Town03)
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
    # It can detect: intersection, t_junction, corridor, highway, roundabout
    supported_topologies: Set[TopologyType] = field(default_factory=lambda: {
        TopologyType.INTERSECTION,
        TopologyType.T_JUNCTION,
        TopologyType.CORRIDOR,
        TopologyType.HIGHWAY,
        TopologyType.ROUNDABOUT,
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
class ValidationRules:
    """Container for category validation macros (stylistic refactor; functional behavior unchanged)."""
    map: MapRequirements = field(default_factory=lambda: MapRequirements(topology=TopologyType.UNKNOWN))
    allow_static_props: bool = True
    # Optional list of pair-agnostic required path relations (no functional enforcement yet).
    required_relations: List["RequiredRelation"] = field(default_factory=list)


@dataclass
class RequiredRelation:
    """
    Declares a simple pairwise path relation that must be present
    (scenario is valid if ANY pair of vehicles satisfies all non-'any' fields).
    """
    entry_relation: Literal[
        "opposite",      # vehicles approach from opposing directions (oncoming)
        "same",          # vehicles share the same approach
        "perpendicular", # approaches roughly 90 degrees apart
        "any",
    ] = "any"

    first_maneuver: EgoManeuver = EgoManeuver.UNKNOWN
    second_maneuver: EgoManeuver = EgoManeuver.UNKNOWN

    exit_relation: Literal[
        "same_exit",     # both exit via the same road/segment
        "different_exit",
        "any",
    ] = "any"

    entry_lane_relation: Literal[
        "same_lane",
        "adjacent_lane",
        "any",
    ] = "any"

    exit_lane_relation: Literal[
        "same_lane",
        "adjacent_lane",
        "merge_into",   # first merges into the lane occupied by second
        "any",
    ] = "any"


@dataclass
class CategoryDefinition:
    """
    Lean scenario category definition for LLM prompt + deterministic validation.
    """
    name: str
    summary: str
    intent: str
    rules: ValidationRules
    must_include: List[str]
    avoid: List[str]
    vary: List[VariationAxis] = field(default_factory=list)
    
    @property
    def map(self) -> MapRequirements:
        return self.rules.map

    @property
    def allow_static_props(self) -> bool:
        return self.rules.allow_static_props


# Honest assessment of each category
CATEGORY_DEFINITIONS: Dict[str, CategoryDefinition] = {
    # Legacy notes/conflict_via kept below each block for reference.
    "Intersection Deadlock Resolution": CategoryDefinition(
        name="Intersection Deadlock Resolution",
        summary="Uncontrolled intersection with multiple approaches and ambiguous right-of-way.",
        intent="Force multi-vehicle negotiation at an uncontrolled intersection where paths cross without clear priority.",
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.INTERSECTION),
            allow_static_props=True,
        ),
        must_include=[
            "Vehicle 1 must be from (entry_road=main) and Vehicle 2 from (entry_road=side)",
            "Each vehicle must participate in at least one semantic maneuver conflict with another vehicle, such as straight-through vs opposing left turn, left turn vs perpendicular straight-through, or right turn merging into an occupied exit lane.",
            "Conflicts must involve overlapping or intersecting planned paths within the intersection or its immediate exits, not merely proximity or sequential yielding.",
            "No vehicle may be behaviorally independent; every maneuver must require negotiation, yielding, or mutual blocking due to another vehicle.",
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
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.INTERSECTION, needs_oncoming=True),
            allow_static_props=False,
            required_relations=[
                RequiredRelation(
                    entry_relation="opposite",
                    first_maneuver=EgoManeuver.LEFT,
                    second_maneuver=EgoManeuver.STRAIGHT,
                )
            ],
        ),
        must_include=[
            "Vehicle 1 must enter from one approach and execute a left turn to a perpendicular exit road.",
            "One vehicle must enter from the opposite approach of Vehicle 1 and continue straight through the intersection (oncoming relative to Vehicle 1).",
            "The left-turn path of Vehicle 1 must geometrically cross the straight-through path of the oncoming vehicle.",
            "Either Vehicle 1 must slow/stop to allow the oncoming vehicle to pass first, or the oncoming vehicle must slow/stop to allow Vehicle 1 to complete the left turn.",
            "Any additional vehicle must interact with Vehicle 1 or the oncoming stream in a meaningful way, such as queueing behind Vehicle 1, following the oncoming vehicle through the intersection, competing for the same exit lane as Vehicle 1, or crossing perpendicularly in a way that constrains the left turn.",
            "No vehicle may traverse the intersection without yielding, slowing, or being constrained by another vehicle.",
        ],
        avoid=[
            "Any static props",
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
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.HIGHWAY, needs_on_ramp=True, needs_merge=True),
            allow_static_props=False,
        ),
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
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.HIGHWAY, needs_multi_lane=True),
            allow_static_props=False,
        ),
        must_include=[
            "All vehicles must begin on a multi-lane highway/corridor with at least two adjacent lanes occupied by different vehicles.",
            "Every vehicle must execute at least one lane change (left or right) during the scenario (no purely lane-keeping vehicles).",
            "Lane changes must occur at different longitudinal positions (staggered along the road), not all at the same point.",
            "For every lane change, the target lane must contain at least one other vehicle at the time of the lane change such that the lane change creates a meaningful interaction (yielding/slowdown/spacing adjustment) with that vehicle.",
            "Lane-change interactions must be persistent: if Vehicle A changes into the lane of Vehicle B, Vehicle B may not have already left that lane before A begins the lane change; the interaction must occur with B while B remains in the target lane.",
            "No lane change may be 'into an empty lane' at the moment it begins; each lane change must be into an occupied lane segment near another vehicle (ahead, alongside, or behind) that constrains the maneuver.",
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
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.CORRIDOR, needs_multi_lane=True),
            allow_static_props=True,
        ),
        must_include=[
            "Static prop fully blocking or partially blocking a lane that a vehicle is travelling in. Only one lane should be blocked.",
            "Vehicle in adjacent lane relative to blocked lane (left/right lane of). Some vehicles may also have a different entry road and turn onto the blocked lane before the blockage.",
            "Additional vehicles may be oncoming relative to the vehicle travelling in the blocked lane, or in the same lane behind the blockage, or in adjacent lanes.",
        ],
        avoid=[
            "Any lane changes"
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
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.CORRIDOR, needs_multi_lane=True, needs_merge=True),
            allow_static_props=True,
        ),
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
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.T_JUNCTION),
            allow_static_props=False,
        ),
        must_include=[
        "ALWAYS: include Vehicle S from side street entering main road.",
        "ALWAYS: include Vehicle M on the main road such that S must yield (S merges into lane of M or merges into occupied exit segment).",
        "WHEN ego_count IN {3,4,5}: you MUST include an oncoming main-road vehicle Vehicle O with constraint OPPOSITE_APPROACH_OF(O, M), going straight through.",
        "WHEN ego_count IN {4,5}: you MUST include a vehicle Vehicle X that turns from the main road into the side road, creating a conflict with S in the junction or immediate exit.",
        ],
        avoid=[
            "Signals controlling entry",
            "ANY lane change by any vehicle",
            "Static props",
        ],
        vary=[
            VariationAxis("ego_count", ["3", "4", "5"], "total vehicles at the T-junction"),
            VariationAxis("main_road_queue", ["single", "follow_route_of chain 2-4"], "volume of main-road traffic"),
            VariationAxis("side_street_queue", ["single", "follow_route_of chain 2"], "volume of side-street entrants"),
            VariationAxis("side_maneuver", ["left turn", "right turn"], "maneuver performed by the side-street vehicle"),
            VariationAxis("pedestrian", ["none", "walker crossing exit path"], "whether a pedestrian crosses the exit path"),
        ],
    ), 

    "Construction Zone": CategoryDefinition(
        name="Construction Zone",
        summary="Corridor work zone with cones/props narrowing lanes.",
        intent="Create forced merges and constrained paths using static props.",
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.INTERSECTION, needs_multi_lane=True),
            allow_static_props=True,
        ),
        must_include=[
            "Define one exit segment, in one vehicles lane to be the construction zone, and do not place any props outside this area",
            "Clusters of multiple types of construction related static props in work zone",
            "Construction props must be selected ONLY from the following assets: cones (constructioncone, trafficcone01, trafficcone02), barriers (streetbarrier, barrel, chainbarrier, chainbarrierend), warning sign (trafficwarning), and at most one construction/utility vehicle (truck or van).",
            "All vehicles exit segments must be either the construction zone lane or an adjacent lane to the construction zone",
            "Vehicles may turn into the exit segment lane"
            
        ],
        avoid=[
            "Pedestrians or cyclists",
            "Any static prop that is not directly behind another static prop in the work zone",
            "Props outside of the lane designated as the construction zone",
            "Unnecessary lane changes"
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4", "5", "6"], "vehicles traversing the work zone"),
            VariationAxis("number_of_prop_types", ["2", "3", "4","5"], "how many different types of construction props are used in the work zone"),
        ],
    ),

    "Pedestrian Crosswalk": CategoryDefinition(
        name="Pedestrian Crosswalk",
        summary="Corridor crossing with pedestrian(s) interacting with vehicles.",
        intent="Test vehicle response to pedestrians crossing, with optional occlusion.",
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.INTERSECTION),
            allow_static_props=False,
        ),
        must_include=[
            "Atleast one walker crossing perpendicular to an ego lane.",
            #"One static occluder (parked_vehicle) positioned ONLY if occlusion is set to parked vehicle blocking view at the right edge of the lane being crossed (lateral = right_edge/half_right), affects_vehicle=null, motion=static, timing_phase=on_approach.",
           # "At least one ego whose path is to the left of the occluder (same road, left of occluder vehicle) so the occluder sits on that ego’s right side.",
        ],
        avoid=[
            #"Any ego or NPC in the lane to the right of the occluder (i.e., no vehicle shares the occluder’s lane).",
            #"Occluder overlapping an ego path or anywhere left of the walker’s crossing lane.",
            #"ANY Static prop besides the occluder.",
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4"], "vehicles approaching the crosswalk"),
            VariationAxis("walker_count", ["1", "2", "3"], "number of pedestrians crossing"),
           #VariationAxis("occlusion", ["none", "parked_vehicle blocking view"], "whether an occluder exists blocking the view of a crossing pedestrian. this occluder must not block the paths of any vehicles"),
            #VariationAxis("occlusion_type", ["box truck", "van", "bus", "delivery truck"], "type of occluding vehicle"),
        ],
    ), 


    "Overtaking on Two-Lane Road": CategoryDefinition(
        name="Overtaking on Two-Lane Road",
        summary="Corridor overtaking/pass maneuvers with adjacent/oncoming/obstacle factors.",
        intent="Exercise overtaking decisions with adjacent lane use, potential oncoming traffic, and obstacles.",
        must_include=[
            "Static prop, such as a parked vehicle, blocking lane in Vehicle 1's path",
            "Vehicle 2 approaches MUST be opposite_approach_as Vehicle 1 (oncoming traffic)",
        ],
        avoid=[
            "Props on either side of the road that do not contribute to the overtaking scenario",
            "Having all vehicles travel in the same direction",
            "Any lane changes",
            "This is a two lane road - no additional lanes may be referenced or used beyond the two lanes (one per direction)",
            "Do not include ANY opposite_approach_as constraint besides between Vehicle 1 and Vehicle 2"
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4"], "vehicles involved in the overtake scenario"),
            VariationAxis("pedestrian", ["none", "walker crossing from sidewalk left or sidewalk right"], "pedestrian involvement during pass"),
        ],
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.TWO_LANE_CORRIDOR, needs_multi_lane=False, needs_oncoming=True),
            allow_static_props=True,
        ),
    ),

    "Roundabout Navigation": CategoryDefinition(
        name="Roundabout Navigation",
        summary="Multi-vehicle negotiation at a roundabout with yielding and merging.",
        intent="Test yield-on-entry and gap acceptance in circular traffic flow with multiple vehicles navigating the roundabout simultaneously.",
        rules=ValidationRules(
            map=MapRequirements(topology=TopologyType.ROUNDABOUT),
            allow_static_props=False,
        ),
        must_include=[
            "At least one pair of vehicles MUST have routes that overlap in the roundabout region (not disjoint). Concretely, for some pair (A,B), at least one of these must be true:",
            "  - Same exit: exit_dir(A) == exit_dir(B) (they end up competing/queueing for the same outbound leg), OR",
            "  - Swap/merge: exit_dir(A) == entry_dir(B) OR exit_dir(B) == entry_dir(A) (one enters where the other is headed), OR",
            "  - Shared travel through the circle: A and B use different exits, but their routes are long enough that they would be in the circle at the same time (not 'enter and immediately exit' for both).",
            "For that overlapping pair, at least one vehicle must slow/yield because of the other. It cannot be two independent drives that just happen to be nearby.",
            "No vehicle may complete its route without being constrained by at least one other vehicle (slowdown, yield, brief stop, or spacing adjustment).",
        ],
        avoid=[
            "Static props of any kind",
            "Pedestrians or cyclists",
            "ANY Lane changes)",
        ],
        vary=[
            VariationAxis("ego_count", ["2", "3", "4", "5"], "vehicles navigating the roundabout"),
        ],
    ),
}


def get_available_categories() -> List[str]:
    """Return all supported categories."""
    return list(CATEGORY_DEFINITIONS.keys())
