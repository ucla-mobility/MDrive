"""
Schema-based scenario generation.

This module produces structured ScenarioSpec JSON instead of free-form text.
"""

import random
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure scenario_generator/ is on sys.path for pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.step_01_crop.llm_utils import _extract_first_json_object

from .capabilities import (
    CATEGORY_DEFINITIONS,
    ActorKind,
    ConstraintType,
    EgoManeuver,
    GroupPattern,
    LateralPosition,
    MotionType,
    SpeedHint,
    TimingPhase,
    TopologyType,
    get_available_categories,
)
from .constraints import (
    EgoVehicleSpec,
    InterVehicleConstraint,
    NonEgoActorSpec,
    ScenarioSpec,
    validate_spec,
    spec_from_dict,
)
from .schema_utils import description_from_spec


_CONSTRAINT_VALUES = [c.value for c in ConstraintType]
_MANEUVER_VALUES = [m.value for m in EgoManeuver]
_TOPOLOGY_VALUES = [t.value for t in TopologyType if t != TopologyType.UNKNOWN]
_ACTOR_KIND_VALUES = [a.value for a in ActorKind]
_GROUP_PATTERN_VALUES = [g.value for g in GroupPattern]
_LATERAL_VALUES = [l.value for l in LateralPosition]
_TIMING_VALUES = [t.value for t in TimingPhase]
_MOTION_VALUES = [m.value for m in MotionType]
_SPEED_VALUES = [s.value for s in SpeedHint]


def select_variation_values(
    cat_info: Any,
    used_combinations: Optional[Set[str]] = None,
) -> Dict[str, str]:
    axes = getattr(cat_info, "vary", []) or []
    used_combinations = used_combinations or set()
    if not axes:
        return {}

    for _ in range(50):
        selections = {axis.name: random.choice(axis.options) for axis in axes if axis.options}
        combo_key = "|".join(f"{k}={v}" for k, v in sorted(selections.items()))
        if combo_key not in used_combinations:
            return selections

    return {axis.name: random.choice(axis.options) for axis in axes if axis.options}


def _enum_list(values: List[str]) -> str:
    return " | ".join(values)


def build_schema_system_prompt() -> str:
    return (
        "You generate structured JSON scenario specs for a driving pipeline.\n"
        "Output ONLY a JSON object that matches the requested schema.\n"
        "Do not include markdown, comments, or extra text."
    )


def build_schema_generation_prompt(
    category: str,
    cat_info: Any,
    forced_variations: Optional[Dict[str, str]] = None,
    previous_validation_feedback: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a compact prompt for schema generation without level-based scaling.
    """
    summary = getattr(cat_info, "summary", "")
    intent = getattr(cat_info, "intent", "")
    must_include = getattr(cat_info, "must_include", []) or []
    avoid = getattr(cat_info, "avoid", []) or []

    topology = cat_info.map.topology if hasattr(cat_info, "map") else TopologyType.UNKNOWN
    needs_oncoming = getattr(cat_info.map, "needs_oncoming", False)
    needs_multi_lane = getattr(cat_info.map, "needs_multi_lane", False)
    needs_on_ramp = getattr(cat_info.map, "needs_on_ramp", False)
    needs_merge = getattr(cat_info.map, "needs_merge", False)

    required_flags = []
    if needs_oncoming:
        required_flags.append("needs_oncoming=true")
    if needs_multi_lane:
        required_flags.append("needs_multi_lane=true")
    if needs_on_ramp:
        required_flags.append("needs_on_ramp=true")
    if needs_merge:
        required_flags.append("needs_merge=true")
    required_flags_str = ", ".join(required_flags) if required_flags else "none"

    variation_lines = []
    axes = getattr(cat_info, "vary", []) or []
    for axis in axes:
        opts = ", ".join(axis.options)
        why = f" (why: {axis.why})" if getattr(axis, "why", "") else ""
        variation_lines.append(f"- {axis.name}: options = {opts}{why}")
    if forced_variations:
        for k, v in forced_variations.items():
            variation_lines.append(f"- {k}: {v}")
    variation_block = "\n".join(variation_lines) if variation_lines else "- (no variation axes defined)"

    schema = f"""  "category": "{category}",
  "topology": "{_enum_list(_TOPOLOGY_VALUES)}",
  "needs_oncoming": true|false,
  "needs_multi_lane": true|false,
  "needs_on_ramp": true|false,
  "needs_merge": true|false,
  "ego_vehicles": [
    {{
      "vehicle_id": "Vehicle 1",
      "maneuver": "{_enum_list(_MANEUVER_VALUES)}",
      "lane_change_phase": "before_intersection" | "after_intersection" | "unknown",
      "entry_road": "main" | "side" | "unknown",
      "exit_road": "main" | "side" | "unknown"
    }}
  ],
  "vehicle_constraints": [
    {{
      "type": "{_enum_list(_CONSTRAINT_VALUES)}",
      "a": "Vehicle X",
      "b": "Vehicle Y"
    }}
  ],
  "actors": [
    {{
      "actor_id": "traffic cones",
      "kind": "{_enum_list(_ACTOR_KIND_VALUES)}",
      "quantity": 1,
      "group_pattern": "{_enum_list(_GROUP_PATTERN_VALUES)}",
      "start_lateral": "{_enum_list(_LATERAL_VALUES)}" | null,
      "end_lateral": "{_enum_list(_LATERAL_VALUES)}" | null,
      "affects_vehicle": "Vehicle 1" | null,
      "timing_phase": "{_enum_list(_TIMING_VALUES)}",
      "lateral_position": "{_enum_list(_LATERAL_VALUES)}",
      "motion": "{_enum_list(_MOTION_VALUES)}",
      "speed": "{_enum_list(_SPEED_VALUES)}",
      "crossing_direction": "left" | "right" | null,
      "direction_relative_to": "same" | "opposite" | null
    }}
  ]
}}"""

    validation_feedback_section = ""
    if previous_validation_feedback:
        score = previous_validation_feedback.get("score", 0.0)
        issues = previous_validation_feedback.get("issues", [])
        if issues:
            error_list = "\n".join(
                f"  - [{issue.get('severity', 'error').upper()}] {issue.get('message', 'Unknown issue')}\n"
                f"    Suggestion: {issue.get('suggestion', 'N/A')}"
                for issue in issues[:10]
            )
            validation_feedback_section = f"""
⚠️  PREVIOUS ATTEMPT FAILED ⚠️
Validation Score: {score:.2f}
Issues found:
{error_list}

Edit the prior JSON directly and fix the problems. Do not invent new keys.
"""

    prompt = (
        f"Generate a JSON scenario spec for the driving pipeline.\n"
        f"Category: {category}\n"
        f"Summary: {summary}\n"
        f"Intent: {intent}\n"
        f"Topology: {topology.value}\n"
        f"Required flags: {required_flags_str}\n"
        "Must include:\n"
        + "\n".join(f"- {item}" for item in must_include) + "\n"
        "Avoid:\n"
        + ("\n".join(f"- {item}" for item in avoid) if avoid else "- none") + "\n"
        f"Variation targets:\n{variation_block}\n"
        "Rules:\n"
        "- Output ONLY JSON (no markdown/comments).\n"
        "- Use 2-5 ego vehicles; IDs must be 'Vehicle 1', 'Vehicle 2', ...\n"
        f"- Allowed constraint types ONLY: {_enum_list(_CONSTRAINT_VALUES)}.\n"
        "- If more than one vehicle, include at least one ACTIVE constraint: perpendicular_left_of, perpendicular_right_of, opposite_approach_of, left_lane_of, right_lane_of, merges_into_lane_of, or same_lane_as.\n"
        "- All constraints must reference existing vehicles; avoid duplicate pairs.\n"
        "- If needs_on_ramp=true: include entry_road='side' AND entry_road='main', plus merges_into_lane_of from side -> main.\n"
        "- If needs_multi_lane or needs_merge=true: include a lane/merge relation (left/right lane, same_lane_as, or merges_into_lane_of).\n"
        "- If needs_oncoming=true: include opposite_approach_of.\n"
        "- Actors are optional; include only if they strengthen the category intent.\n"
        f"{validation_feedback_section}Return ONLY a JSON object:\n"
        "{\n"
        '  "scenario_spec": {\n'
        f"{schema}\n"
        "  }\n"
        "}\n"
    )
    return prompt


def build_schema_repair_prompt(
    bad_payload: str,
    errors: List[str],
    category: str,
    cat_info: Any,
) -> str:
    valid_constraint_types = _enum_list(_CONSTRAINT_VALUES)
    error_lines = "\n".join(f"- {e}" for e in errors) if errors else "- (unknown errors)"
    required_flags = []
    if cat_info.map.needs_oncoming:
        required_flags.append("oncoming")
    if cat_info.map.needs_multi_lane:
        required_flags.append("multi_lane")
    if cat_info.map.needs_on_ramp:
        required_flags.append("on_ramp")
    if cat_info.map.needs_merge:
        required_flags.append("merge")
    flags_line = ", ".join(required_flags) if required_flags else "none"

    critical_reminders = [
        "- If more than one vehicle: include at least one ACTIVE constraint (perpendicular/opposite/lane/merge/same_lane_as).",
        "- All constraints must reference existing vehicles; no duplicates.",
    ]
    if cat_info.map.needs_on_ramp:
        critical_reminders.extend([
            "- Include entry_road='main' for at least one vehicle (mainline).",
            "- Include entry_road='side' for at least one vehicle (on-ramp).",
            "- Include merges_into_lane_of from side vehicle to main vehicle.",
        ])
    if cat_info.map.needs_multi_lane or cat_info.map.needs_merge:
        critical_reminders.append("- Include a lane/merge relation: left_lane_of/right_lane_of/same_lane_as/merges_into_lane_of.")

    reminders_block = "\n".join(critical_reminders)
    
    return (
        "Fix the JSON scenario spec to satisfy the errors below. Edit the prior JSON directly; change only what is needed.\n"
        f"Category: {category}\n"
        f"Required topology: {cat_info.map.topology.value}\n"
        f"Required flags: {flags_line}\n"
        "\n"
        "CRITICAL REMINDERS:\n"
        f"{reminders_block}\n"
        "\n"
        "ERRORS TO FIX:\n"
        f"{error_lines}\n"
        "\n"
        "CONSTRAINT RULES:\n"
        f"- Allowed constraint types: {valid_constraint_types}\n"
        "- DO NOT invent new constraint types or keys. Only use the allowed values above.\n"
        "- If more than one vehicle: include at least one ACTIVE constraint (perpendicular/opposite/lane/merge/same_lane_as).\n"
        "\n"
        "YOUR PREVIOUS ATTEMPT (failed validation) — edit THIS JSON:\n"
        f"{bad_payload}\n"
        "\n"
        "Return ONLY a corrected JSON object with:\n"
        "1. All required fields properly set (especially entry_road for on-ramp scenarios)\n"
        "2. At least one active constraint when multiple vehicles are present\n"
        "3. Valid vehicle IDs and constraint references\n"
        "4. Reuse vehicle IDs unless an error explicitly requires adding/removing vehicles.\n"
    )


def _actor_conflict_info(spec: ScenarioSpec) -> Tuple[bool, Set[str], bool]:
    has_conflict = False
    actor_targets: Set[str] = set()
    affects_all = False

    for actor in spec.actors:
        if actor.affects_vehicle:
            actor_targets.add(actor.affects_vehicle)

        if actor.motion == MotionType.CROSS_PERPENDICULAR and actor.kind in {ActorKind.WALKER, ActorKind.CYCLIST}:
            has_conflict = True
            affects_all = True
            continue

        if actor.motion == MotionType.STATIC and actor.kind in {ActorKind.PARKED_VEHICLE, ActorKind.STATIC_PROP}:
            if actor.affects_vehicle or actor.lateral_position != LateralPosition.CENTER:
                has_conflict = True

    return has_conflict, actor_targets, affects_all


def _conflict_findings(spec: ScenarioSpec, cat_info: Any, relaxed: bool = False) -> Tuple[List[str], List[str]]:
    """
    Lightweight structural checks to keep generated specs pipeline-friendly.
    Enforces allowed references and category-required flags without level-based scaling.
    """
    errors: List[str] = []
    warnings: List[str] = []

    vehicles = [v.vehicle_id for v in spec.ego_vehicles]
    num_vehicles = len(vehicles)

    passive_constraint_types = {
        ConstraintType.FOLLOW_ROUTE_OF,
        ConstraintType.SAME_APPROACH_AS,
    }
    active_constraint_types = {
        ConstraintType.OPPOSITE_APPROACH_OF,
        ConstraintType.PERPENDICULAR_LEFT_OF,
        ConstraintType.PERPENDICULAR_RIGHT_OF,
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
        ConstraintType.SAME_LANE_AS,
    }
    lane_types = {
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
        ConstraintType.SAME_LANE_AS,
    }
    crossing_types = {
        ConstraintType.OPPOSITE_APPROACH_OF,
        ConstraintType.PERPENDICULAR_LEFT_OF,
        ConstraintType.PERPENDICULAR_RIGHT_OF,
    }

    active_constraints = [c for c in spec.vehicle_constraints if c.constraint_type in active_constraint_types]

    if num_vehicles > 1 and not active_constraints:
        errors.append(
            "Include at least one ACTIVE constraint (perpendicular/opposite/lane/merge/same_lane_as) for multi-vehicle specs."
        )

    topology = cat_info.map.topology
    if topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}:
        if not any(c.constraint_type in crossing_types for c in spec.vehicle_constraints):
            warnings.append("Intersection-like categories should include a crossing relation (perpendicular or opposite).")
        if not cat_info.map.needs_on_ramp:
            known_entries = [v.entry_road for v in spec.ego_vehicles if v.entry_road in {"main", "side"}]
            distinct_entries = set(known_entries)
            if num_vehicles >= 2 and len(distinct_entries) <= 1:
                errors.append(
                    "Intersection-like categories require vehicles from multiple approaches: mix entry_road values (main vs side)."
                )
            elif num_vehicles >= 2 and not known_entries:
                warnings.append(
                    "Set entry_road explicitly (main/side) so vehicles come from different approaches in intersection-like categories."
                )

    if cat_info.map.needs_oncoming:
        if not any(c.constraint_type == ConstraintType.OPPOSITE_APPROACH_OF for c in spec.vehicle_constraints):
            errors.append("Category requires oncoming traffic: include opposite_approach_of.")

    if cat_info.map.needs_multi_lane:
        if not any(c.constraint_type in lane_types for c in spec.vehicle_constraints):
            errors.append("Category requires multi-lane relation: include left/right lane, same_lane_as, or merges_into_lane_of.")

    if cat_info.map.needs_merge or cat_info.map.needs_on_ramp:
        if not any(c.constraint_type == ConstraintType.MERGES_INTO_LANE_OF for c in spec.vehicle_constraints):
            errors.append("Category requires merge conflict: include merges_into_lane_of.")

    if cat_info.map.needs_on_ramp:
        entry_by_vehicle = {v.vehicle_id: v.entry_road for v in spec.ego_vehicles}
        side_vehicles = [v for v, entry in entry_by_vehicle.items() if entry == "side"]
        main_vehicles = [v for v, entry in entry_by_vehicle.items() if entry == "main"]
        if not side_vehicles:
            errors.append("On-ramp scenarios require at least one vehicle with entry_road='side'.")
        if not main_vehicles:
            errors.append("On-ramp scenarios require at least one vehicle with entry_road='main'.")
        if any(c.constraint_type == ConstraintType.MERGES_INTO_LANE_OF for c in spec.vehicle_constraints):
            valid_merge = False
            for c in spec.vehicle_constraints:
                if c.constraint_type != ConstraintType.MERGES_INTO_LANE_OF:
                    continue
                a_entry = entry_by_vehicle.get(c.vehicle_a, "unknown")
                b_entry = entry_by_vehicle.get(c.vehicle_b, "unknown")
                if a_entry == "side" and b_entry == "main":
                    valid_merge = True
                    break
            if not valid_merge:
                warnings.append("Use merges_into_lane_of from side -> main for on-ramp scenarios.")

    if num_vehicles > 1:
        covered: Set[str] = set()
        for c in spec.vehicle_constraints:
            covered.add(c.vehicle_a)
            covered.add(c.vehicle_b)
        actor_targets: Set[str] = set()
        for actor in spec.actors:
            if actor.affects_vehicle:
                actor_targets.add(actor.affects_vehicle)
        covered.update(actor_targets)
        uncovered = [v for v in vehicles if v not in covered]
        if uncovered:
            warnings.append(
                f"{', '.join(uncovered)} not referenced by any constraint or actor; ensure every vehicle participates in an interaction."
            )

    # Warn if only passive constraints exist
    if spec.vehicle_constraints and all(c.constraint_type in passive_constraint_types for c in spec.vehicle_constraints):
        warnings.append("All constraints are passive; add at least one active relation for interaction.")

    return errors, warnings


def _ensure_direction_and_lane_constraints(spec: ScenarioSpec) -> ScenarioSpec:
    """
    Auto-fix missing basics for multi-vehicle interaction categories.
    Ensures on-ramp merge relations, lane/merge constraints, and directional references,
    but does not overwrite existing vehicle entries.
    """
    if spec.needs_on_ramp and len(spec.ego_vehicles) < 2:
        new_id = f"Vehicle {len(spec.ego_vehicles) + 1}"
        spec.ego_vehicles.append(
            EgoVehicleSpec(
                vehicle_id=new_id,
                maneuver=EgoManeuver.LANE_CHANGE,
                lane_change_phase="unknown",
                entry_road="side",
                exit_road="main",
            )
        )

    if len(spec.ego_vehicles) < 2:
        return spec

    actor_conflict, _, _ = _actor_conflict_info(spec)
    vehicle_b = spec.ego_vehicles[0].vehicle_id
    vehicle_a = spec.ego_vehicles[1].vehicle_id

    existing_types = {c.constraint_type for c in spec.vehicle_constraints}
    existing_keys = {(c.constraint_type, c.vehicle_a, c.vehicle_b) for c in spec.vehicle_constraints}
    crossing_types = {
        ConstraintType.OPPOSITE_APPROACH_OF,
        ConstraintType.PERPENDICULAR_LEFT_OF,
        ConstraintType.PERPENDICULAR_RIGHT_OF,
    }
    lane_types = {
        ConstraintType.LEFT_LANE_OF,
        ConstraintType.RIGHT_LANE_OF,
        ConstraintType.MERGES_INTO_LANE_OF,
    }

    def add_constraint(constraint_type: ConstraintType, a: str, b: str) -> None:
        key = (constraint_type, a, b)
        if key in existing_keys:
            return
        spec.vehicle_constraints.append(
            InterVehicleConstraint(
                constraint_type=constraint_type,
                vehicle_a=a,
                vehicle_b=b,
            )
        )
        existing_keys.add(key)
        existing_types.add(constraint_type)

    # Normalize on-ramp entries and enforce merge constraint without clobbering existing assignments
    if spec.needs_on_ramp:
        main_vehicle = next((v for v in spec.ego_vehicles if v.entry_road == "main"), spec.ego_vehicles[0])
        if main_vehicle.entry_road == "unknown":
            main_vehicle.entry_road = "main"
            main_vehicle.exit_road = "main"
        side_vehicle = next((v for v in spec.ego_vehicles if v.entry_road == "side"), None)
        if side_vehicle is None:
            candidates = [v for v in spec.ego_vehicles if v.vehicle_id != main_vehicle.vehicle_id]
            side_vehicle = candidates[0] if candidates else spec.ego_vehicles[1]
            if side_vehicle.entry_road == "unknown":
                side_vehicle.entry_road = "side"
            if side_vehicle.exit_road == "unknown":
                side_vehicle.exit_road = "main"
            if side_vehicle.maneuver == EgoManeuver.STRAIGHT:
                side_vehicle.maneuver = EgoManeuver.LANE_CHANGE
        add_constraint(ConstraintType.MERGES_INTO_LANE_OF, side_vehicle.vehicle_id, main_vehicle.vehicle_id)

    # For intersection/T junction categories without on-ramp, ensure at least two approaches are represented
    if spec.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION} and not spec.needs_on_ramp:
        first = spec.ego_vehicles[0]
        second = spec.ego_vehicles[1]
        if first.entry_road == "unknown":
            first.entry_road = "main"
        if second.entry_road == "unknown" or second.entry_road == first.entry_road:
            second.entry_road = "side"
        if second.exit_road == "unknown":
            second.exit_road = "unknown" if spec.topology == TopologyType.INTERSECTION else "main"

    if spec.needs_merge and not spec.needs_on_ramp:
        add_constraint(ConstraintType.MERGES_INTO_LANE_OF, vehicle_a, vehicle_b)

    if spec.needs_oncoming:
        add_constraint(ConstraintType.OPPOSITE_APPROACH_OF, vehicle_a, vehicle_b)

    if spec.needs_merge or spec.needs_on_ramp:
        add_constraint(ConstraintType.MERGES_INTO_LANE_OF, vehicle_a, vehicle_b)

    if spec.needs_multi_lane and not any(c.constraint_type in lane_types for c in spec.vehicle_constraints):
        # Only inject lane adjacency when vehicles plausibly share a corridor/highway or the same approach
        same_entry = spec.ego_vehicles[0].entry_road == spec.ego_vehicles[1].entry_road and spec.ego_vehicles[0].entry_road != "unknown"
        if spec.topology in {TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY} or same_entry:
            add_constraint(ConstraintType.LEFT_LANE_OF, vehicle_a, vehicle_b)

    if (
        spec.topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION}
        and not any(c.constraint_type in crossing_types for c in spec.vehicle_constraints)
        and not spec.needs_oncoming
    ):
        add_constraint(ConstraintType.PERPENDICULAR_RIGHT_OF, vehicle_a, vehicle_b)

    if actor_conflict or spec.topology in {TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY}:
        direction_types = {
            ConstraintType.SAME_APPROACH_AS,
            ConstraintType.OPPOSITE_APPROACH_OF,
            ConstraintType.PERPENDICULAR_LEFT_OF,
            ConstraintType.PERPENDICULAR_RIGHT_OF,
            ConstraintType.LEFT_LANE_OF,
            ConstraintType.RIGHT_LANE_OF,
            ConstraintType.MERGES_INTO_LANE_OF,
            ConstraintType.FOLLOW_ROUTE_OF,
        }
        # Get the entry_road of the reference vehicle (vehicle_b = ego_vehicles[0])
        ref_entry_road = spec.ego_vehicles[0].entry_road
        for vehicle in spec.ego_vehicles[1:]:
            has_direction = any(
                c.vehicle_a == vehicle.vehicle_id and c.constraint_type in direction_types
                for c in spec.vehicle_constraints
            )
            if not has_direction:
                # Only add follow_route_of if vehicles share the same entry_road
                # (or both are unknown). Different entry_roads imply different approach paths.
                if vehicle.entry_road == ref_entry_road or vehicle.entry_road == "unknown" or ref_entry_road == "unknown":
                    add_constraint(ConstraintType.FOLLOW_ROUTE_OF, vehicle.vehicle_id, vehicle_b)

    return spec


@dataclass
class SchemaGenerationConfig:
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    max_retries: int = 3
    allow_template_fallback: bool = False


class TemplateSchemaGenerator:
    """Deterministic schema generator for quick testing."""

    def generate_spec(self, category: str) -> ScenarioSpec:
        cat_info = CATEGORY_DEFINITIONS[category]
        topology = cat_info.map.topology
        needs_oncoming = cat_info.map.needs_oncoming
        needs_multi_lane = cat_info.map.needs_multi_lane
        needs_on_ramp = cat_info.map.needs_on_ramp
        needs_merge = cat_info.map.needs_merge

        ego_count = 3 if (needs_merge or needs_on_ramp or needs_multi_lane) else 2

        vehicles: List[EgoVehicleSpec] = []
        constraints: List[InterVehicleConstraint] = []

        base_maneuver = EgoManeuver.LANE_CHANGE if (needs_merge or needs_on_ramp) else EgoManeuver.STRAIGHT
        if topology in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION} and not needs_merge:
            base_maneuver = EgoManeuver.STRAIGHT

        vehicles.append(
            EgoVehicleSpec(
                vehicle_id="Vehicle 1",
                maneuver=base_maneuver,
                lane_change_phase="after_intersection" if base_maneuver == EgoManeuver.LANE_CHANGE and topology not in {TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY} else "unknown",
                entry_road="main" if needs_on_ramp else "unknown",
                exit_road="main" if needs_on_ramp else "unknown",
            )
        )

        vehicles.append(
            EgoVehicleSpec(
                vehicle_id="Vehicle 2",
                maneuver=EgoManeuver.LANE_CHANGE if (needs_merge or needs_on_ramp or needs_multi_lane) else EgoManeuver.STRAIGHT,
                lane_change_phase="after_intersection" if topology not in {TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY} else "unknown",
                entry_road="side" if needs_on_ramp else "unknown",
                exit_road="main" if needs_on_ramp else "unknown",
            )
        )

        if ego_count >= 3:
            vehicles.append(
                EgoVehicleSpec(
                    vehicle_id="Vehicle 3",
                    maneuver=EgoManeuver.STRAIGHT,
                    lane_change_phase="unknown",
                    entry_road="main" if needs_on_ramp else "unknown",
                    exit_road="main" if needs_on_ramp else "unknown",
                )
            )

        actors: List[NonEgoActorSpec] = []

        spec = ScenarioSpec(
            category=category,
            topology=topology,
            needs_oncoming=needs_oncoming,
            needs_multi_lane=needs_multi_lane,
            needs_on_ramp=needs_on_ramp,
            needs_merge=needs_merge,
            ego_vehicles=vehicles,
            vehicle_constraints=constraints,
            actors=actors,
        )
        spec = _ensure_direction_and_lane_constraints(spec)
        spec.description = description_from_spec(spec)
        return spec


class SchemaScenarioGenerator:
    """
    LLM-backed schema generator with validation and optional template fallback.
    """

    def __init__(
        self,
        config: Optional[SchemaGenerationConfig] = None,
        model=None,
        tokenizer=None,
        template_only: bool = False,
    ):
        self.config = config or SchemaGenerationConfig()
        self.template_only = template_only
        self._model = model
        self._tokenizer = tokenizer
        self._template = TemplateSchemaGenerator()
        self.available_categories = get_available_categories()
        self.used_combinations: Dict[str, Set[str]] = {c: set() for c in self.available_categories}
        self.generated_signatures: Dict[str, Set[str]] = {c: set() for c in self.available_categories}

    def _load_model(self):
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()

    def _generate_text(self, prompt: str, temperature_override: Optional[float] = None) -> str:
        self._load_model()
        import torch

        messages = [
            {"role": "system", "content": build_schema_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(text, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        do_sample = bool(self.config.do_sample)
        effective_temp = temperature_override if temperature_override is not None else self.config.temperature
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": effective_temp if do_sample else 0.0,
            "top_p": self.config.top_p if do_sample else 1.0,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
            "repetition_penalty": self.config.repetition_penalty,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response

    def _normalize_schema_obj(
        self,
        obj: Dict[str, Any],
        category: str,
        cat_info: Any,
    ) -> Dict[str, Any]:
        # Phase 1: Extract reasoning if present (new format: {"reasoning": {...}, "scenario_spec": {...}})
        reasoning = None
        if "reasoning" in obj and "scenario_spec" in obj:
            reasoning = obj.get("reasoning", {})
            obj = obj.get("scenario_spec", {})
        # If repair format uses {"repair_of": {...}, "scenario_spec": {...}}, keep scenario_spec
        if "repair_of" in obj and "scenario_spec" in obj:
            obj = obj.get("scenario_spec", {})
        # If plain {"scenario_spec": {...}} format, unwrap it
        if "scenario_spec" in obj and isinstance(obj.get("scenario_spec"), dict):
            obj = obj.get("scenario_spec", {})
        
        out = dict(obj) if isinstance(obj, dict) else {}
        
        # Phase 1: Store reasoning in metadata for validation/debugging
        if reasoning:
            out["_reasoning"] = reasoning
        
        out["category"] = category
        out["topology"] = cat_info.map.topology.value
        # Clamp capability flags to the category definition; do not let the model opt into other geometry classes
        out["needs_oncoming"] = bool(cat_info.map.needs_oncoming)
        out["needs_multi_lane"] = bool(cat_info.map.needs_multi_lane)
        out["needs_on_ramp"] = bool(cat_info.map.needs_on_ramp)
        out["needs_merge"] = bool(cat_info.map.needs_merge or cat_info.map.needs_on_ramp)

        vehicles = out.get("ego_vehicles")
        if not isinstance(vehicles, list) or not vehicles:
            vehicles = [{"vehicle_id": "Vehicle 1", "maneuver": "straight", "lane_change_phase": "unknown",
                         "entry_road": "unknown", "exit_road": "unknown"}]
        for idx, v in enumerate(vehicles, start=1):
            if not isinstance(v, dict):
                vehicles[idx - 1] = {
                    "vehicle_id": f"Vehicle {idx}",
                    "maneuver": "straight",
                    "lane_change_phase": "unknown",
                    "entry_road": "unknown",
                    "exit_road": "unknown",
                }
                continue
            v.setdefault("vehicle_id", f"Vehicle {idx}")
            v.setdefault("maneuver", "straight")
            v.setdefault("lane_change_phase", "unknown")
            v.setdefault("entry_road", "unknown")
            v.setdefault("exit_road", "unknown")
        vehicle_ids = {v.get("vehicle_id") for v in vehicles if isinstance(v, dict)}
        if "Vehicle 1" not in vehicle_ids:
            vehicles.insert(0, {
                "vehicle_id": "Vehicle 1",
                "maneuver": "straight",
                "lane_change_phase": "unknown",
                "entry_road": "unknown",
                "exit_road": "unknown",
            })
        out["ego_vehicles"] = vehicles

        constraints = out.get("vehicle_constraints")
        if not isinstance(constraints, list):
            constraints = []
        out["vehicle_constraints"] = constraints

        actors = out.get("actors")
        if not isinstance(actors, list):
            actors = []
        out["actors"] = actors

        return out

    def _signature(self, spec: ScenarioSpec) -> str:
        vehicle_sig = ",".join(f"{v.vehicle_id}:{v.maneuver.value}" for v in spec.ego_vehicles)
        constraint_sig = ",".join(f"{c.constraint_type.value}:{c.vehicle_a}->{c.vehicle_b}" for c in spec.vehicle_constraints)
        actor_sig = ",".join(f"{a.kind.value}:{a.quantity}" for a in spec.actors)
        return "|".join([vehicle_sig, constraint_sig, actor_sig])

    def generate_spec(
        self,
        category: str,
        stats: Optional[Dict[str, Any]] = None,
        exclude_signatures: Optional[Set[str]] = None,
        previous_validation_feedback: Optional[Dict[str, Any]] = None,
        debug_dir: Optional[Path] = None,
    ) -> Tuple[Optional[ScenarioSpec], List[str], List[str]]:
        """Generate a scenario spec.
        
        Args:
            category: Scenario category
            stats: Optional dict to populate with generation statistics
            exclude_signatures: Optional set of spec signatures to avoid (for retry attempts)
            previous_validation_feedback: Optional dict with validation results from failed attempt:
                {"score": float, "issues": [{"severity": str, "message": str, ...}]}
            debug_dir: Optional path to write debug artifacts (prompts/responses/normalized specs)
        
        Returns:
            Tuple of (spec, errors, warnings). spec is None on failure.
        """
        if category not in self.available_categories:
            return None, [f"Category '{category}' is not supported"], []

        cat_info = CATEGORY_DEFINITIONS[category]
        if self.template_only:
            return self._template.generate_spec(category), [], []

        exclude_signatures = exclude_signatures or set()
        used = self.used_combinations.get(category, set())
        forced_variations = select_variation_values(cat_info, used)
        combo_key = "|".join(f"{k}={v}" for k, v in sorted(forced_variations.items()))
        debug_path = Path(debug_dir) if debug_dir else None
        if debug_path:
            debug_path.mkdir(parents=True, exist_ok=True)

        # Derive banned constraints from previous validation feedback (INEFFECTIVE hints)
        banned_constraints: Set[str] = set()
        if previous_validation_feedback:
            for issue in previous_validation_feedback.get("issues", []):
                text = " ".join([
                    issue.get("message", "") or "",
                    issue.get("suggestion", "") or "",
                ])
                import re
                for m in re.findall(r'(\w+)\(Vehicle\s*(\d+)\s*->\s*Vehicle\s*(\d+)\)', text):
                    banned_constraints.add(f"{m[0].lower()}(Vehicle {m[1]} -> Vehicle {m[2]})")

        prompt = build_schema_generation_prompt(
            category, cat_info, forced_variations, 
            previous_validation_feedback=previous_validation_feedback,
        )
        last_errors: List[str] = []
        last_warnings: List[str] = []
        last_payload = ""

        attempt_count = 0
        repair_attempts = 0
        fallback_spec: Optional[ScenarioSpec] = None
        fallback_warnings: List[str] = []
        # Use higher temperature for retries to encourage exploration
        is_outer_retry = previous_validation_feedback is not None
        retry_temp_boost = 0.2 if is_outer_retry else 0.0  # Boost temp by 0.2 on outer retries
        for attempt in range(self.config.max_retries):
            attempt_count += 1
            if attempt > 0:
                repair_attempts += 1
            # Further boost temperature on inner repair attempts
            inner_temp_boost = 0.1 * attempt  # +0.1 per repair attempt
            # For repair attempts, keep temperature modest to reduce hallucinated constraint types
            base_temp = self.config.temperature + retry_temp_boost + inner_temp_boost
            effective_temp = min(0.8 if attempt > 0 else 1.0, base_temp)
            if debug_path:
                (debug_path / f"schema_prompt_attempt{attempt + 1}.txt").write_text(prompt, encoding="utf-8")
            response = self._generate_text(prompt, temperature_override=effective_temp)
            last_payload = response
            if debug_path:
                (debug_path / f"schema_raw_response_attempt{attempt + 1}.txt").write_text(response, encoding="utf-8")
            obj = _extract_first_json_object(response)
            if obj is None:
                last_errors = ["No valid JSON object found"]
                last_warnings = []
            else:
                normalized = self._normalize_schema_obj(obj, category, cat_info)
                if debug_path:
                    try:
                        (debug_path / f"schema_normalized_attempt{attempt + 1}.json").write_text(
                            json.dumps(normalized, indent=2), encoding="utf-8"
                        )
                    except Exception:
                        pass
                try:
                    spec = spec_from_dict(normalized)
                except Exception as exc:
                    last_errors = [f"Schema parse error: {exc}"]
                    last_warnings = []
                    # If parse failed due to invalid constraint types, tighten the prompt for next attempt
                    invalid_constraints = []
                    msg = str(exc)
                    if "ConstraintType" in msg:
                        invalid_constraints.append(msg)
                    if invalid_constraints and attempt < self.config.max_retries - 1:
                        prompt = build_schema_repair_prompt(last_payload, invalid_constraints, category, cat_info)
                else:
                    spec = _ensure_direction_and_lane_constraints(spec)
                    valid, errors = validate_spec(spec)
                    conflict_errors, conflict_warnings = _conflict_findings(spec, cat_info)
                    if conflict_errors:
                        errors = errors + conflict_errors
                        valid = False
                    last_warnings = conflict_warnings
                    # Reject specs that reuse banned constraints from previous feedback
                    if valid:
                        constraint_signatures = {
                            f"{c.constraint_type.value}(Vehicle {c.vehicle_a.split()[-1]} -> Vehicle {c.vehicle_b.split()[-1]})"
                            for c in spec.vehicle_constraints
                        }
                        if banned_constraints and any(sig.lower() in banned_constraints for sig in constraint_signatures):
                            valid = False
                            errors = [f"Spec reuses banned constraints: {sorted(banned_constraints)}"]
                    if valid:
                        spec.description = description_from_spec(spec)
                        sig = self._signature(spec)
                        duplicate_sig = sig in self.generated_signatures.get(category, set())
                        failed_sig = exclude_signatures and sig in exclude_signatures
                        if duplicate_sig or failed_sig:
                            fallback_spec = spec
                            fallback_warnings = conflict_warnings
                            self.used_combinations[category].add(combo_key)
                            last_errors = ["Generated spec duplicates a previous scenario - retrying"]
                            last_warnings = conflict_warnings
                            valid = False
                        else:
                            self.used_combinations[category].add(combo_key)
                            self.generated_signatures[category].add(sig)
                            if stats is not None:
                                stats["schema_generation_attempts"] = attempt_count
                                stats["schema_generation_repair_attempts"] = repair_attempts
                                stats["schema_template_fallback"] = 0
                            return spec, [], conflict_warnings
                    else:
                        last_errors = errors

            if attempt < self.config.max_retries - 1:
                prompt = build_schema_repair_prompt(last_payload, last_errors, category, cat_info)

        if stats is not None:
            stats["schema_generation_attempts"] = attempt_count
            stats["schema_generation_repair_attempts"] = repair_attempts
            stats["schema_template_fallback"] = 0
        if fallback_spec is not None:
            return fallback_spec, [], fallback_warnings
        if debug_path:
            summary = {
                "errors": last_errors,
                "warnings": last_warnings,
                "attempts": attempt_count,
            }
            try:
                (debug_path / "schema_failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            except Exception:
                pass
        return None, last_errors, last_warnings
