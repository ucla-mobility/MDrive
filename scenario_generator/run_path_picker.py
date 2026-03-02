#!/usr/bin/env python3
"""
run_path_picker.py

Wrapper for pipeline/step_03_path_picker.
"""

from pipeline.step_03_path_picker.candidates import (
    _build_road_corridors,
    _candidate_all_road_ids,
    _candidate_entry_cardinal,
    _candidate_entry_heading_rad,
    _candidate_entry_in_crop,
    _candidate_entry_lane_id,
    _candidate_entry_point,
    _candidate_entry_road_id,
    _candidate_exit_cardinal,
    _candidate_exit_lane_id,
    _candidate_exit_road_id,
    _candidate_geometric_discontinuity,
    _candidate_lane_change_count,
    _candidate_lane_change_phase,
    _candidate_lane_inconsistency,
    _candidate_length,
    _candidate_straight_lateral_drift,
)
from pipeline.step_03_path_picker.constraints import (
    _ALLOWED_CONSTRAINT_TYPES,
    _build_constraints_prompt,
    _candidate_matches_unary,
    _detect_npc_vehicles,
    _filter_constraints_with_evidence,
    _infer_road_role_sets,
    _normalize_constraints_obj,
    _opposite_cardinal,
    _sanitize_constraints_obj,
)
from pipeline.step_03_path_picker.csp import _consistent_with_constraints, _solve_paths_csp
from pipeline.step_03_path_picker.fuzzy import _best_fuzzy_match, _normalize
from pipeline.step_03_path_picker.main import main, pick_paths_with_model
from pipeline.step_03_path_picker.parsing import (
    _extract_description_from_prompt,
    _extract_first_json_object,
    _extract_json_from_codeblocks,
    _extract_vehicles_loose,
    _safe_parse_json_object,
    _safe_parse_model_output,
)
from pipeline.step_03_path_picker.viz import (
    _build_segments_minimal,
    _load_nodes,
    _plot_paths_together,
    _wrap180,
    plt,
)

__all__ = [
    "_ALLOWED_CONSTRAINT_TYPES",
    "_best_fuzzy_match",
    "_build_constraints_prompt",
    "_build_road_corridors",
    "_build_segments_minimal",
    "_candidate_all_road_ids",
    "_candidate_entry_cardinal",
    "_candidate_entry_heading_rad",
    "_candidate_entry_in_crop",
    "_candidate_entry_lane_id",
    "_candidate_entry_point",
    "_candidate_entry_road_id",
    "_candidate_exit_cardinal",
    "_candidate_exit_lane_id",
    "_candidate_exit_road_id",
    "_candidate_geometric_discontinuity",
    "_candidate_lane_change_count",
    "_candidate_lane_change_phase",
    "_candidate_lane_inconsistency",
    "_candidate_length",
    "_candidate_matches_unary",
    "_candidate_straight_lateral_drift",
    "_consistent_with_constraints",
    "_detect_npc_vehicles",
    "_extract_description_from_prompt",
    "_extract_first_json_object",
    "_extract_json_from_codeblocks",
    "_extract_vehicles_loose",
    "_filter_constraints_with_evidence",
    "_infer_road_role_sets",
    "_load_nodes",
    "_normalize",
    "_normalize_constraints_obj",
    "_opposite_cardinal",
    "_plot_paths_together",
    "_safe_parse_json_object",
    "_safe_parse_model_output",
    "_sanitize_constraints_obj",
    "_solve_paths_csp",
    "_wrap180",
    "main",
    "pick_paths_with_model",
    "plt",
]


if __name__ == "__main__":
    main()
