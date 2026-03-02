"""
Scenario Generator Package

Automated generation of diverse, valid driving scenarios for multi-agent coordination testing.
"""

from .capabilities import (
    CATEGORY_DEFINITIONS,
    CategoryDefinition,
    MapRequirements,
    ValidationRules,
    RequiredRelation,
    VariationAxis,
    TopologyType,
    ActorKind,
    MotionType,
    ConstraintType,
    EgoManeuver,
    LateralPosition,
    TimingPhase,
    get_available_categories,
)
from .constraints import (
    ScenarioSpec,
    EgoVehicleSpec,
    InterVehicleConstraint,
    NonEgoActorSpec,
    validate_spec,
    spec_to_dict,
    spec_from_dict,
)
from .scene_validator import SceneValidator, SceneValidationResult, ValidationIssue, ValidationIssueDetail
from .pipeline_runner import PipelineRunner, GenerationLogger
from .schema_utils import (
    geometry_spec_from_scenario_spec,
    description_from_spec,
)
from .schema_generator import (
    SchemaScenarioGenerator,
    SchemaGenerationConfig,
    TemplateSchemaGenerator,
)
from .schema_generation_loop import (
    SchemaGenerationLoop,
    SchemaGenerationLoopConfig,
    SchemaGenerationResult,
    DEFAULT_SCHEMA_CATEGORIES,
)

__all__ = [
    # Capabilities
    "CATEGORY_DEFINITIONS",
    "CategoryDefinition",
    "MapRequirements",
    "ValidationRules",
    "RequiredRelation",
    "VariationAxis",
    "TopologyType",
    "ActorKind",
    "MotionType", 
    "ConstraintType",
    "EgoManeuver",
    "LateralPosition",
    "TimingPhase",
    "get_available_categories",
    # Constraints
    "ScenarioSpec",
    "EgoVehicleSpec",
    "InterVehicleConstraint",
    "NonEgoActorSpec",
    "validate_spec",
    "spec_to_dict",
    "spec_from_dict",
    # Scene Validator
    "SceneValidator",
    "SceneValidationResult",
    "ValidationIssue",
    "ValidationIssueDetail",
    # Pipeline Runner
    "PipelineRunner",
    "GenerationLogger",
    # Schema utils + generator
    "geometry_spec_from_scenario_spec",
    "description_from_spec",
    "SchemaScenarioGenerator",
    "SchemaGenerationConfig",
    "TemplateSchemaGenerator",
    "SchemaGenerationLoop",
    "SchemaGenerationLoopConfig",
    "SchemaGenerationResult",
    "DEFAULT_SCHEMA_CATEGORIES",
]
