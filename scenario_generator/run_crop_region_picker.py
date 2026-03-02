#!/usr/bin/env python3
"""
run_crop_region_picker.py

Wrapper for pipeline/step_01_crop.
"""

from pipeline.step_01_crop.candidates import build_candidate_crops_for_town, build_corridor_candidate_crops_for_town, build_roundabout_crop_for_town
from pipeline.step_01_crop.csp import solve_assignment
from pipeline.step_01_crop.llm_extractor import LLMGeometryExtractor
from pipeline.step_01_crop.main import main
from pipeline.step_01_crop.models import CropFeatures, CropKey, GeometrySpec, Scenario
from pipeline.step_01_crop.scenario_io import load_scenarios
from pipeline.step_01_crop.viz import plt, mpatches

__all__ = [
    "CropFeatures",
    "CropKey",
    "GeometrySpec",
    "LLMGeometryExtractor",
    "Scenario",
    "build_candidate_crops_for_town",
    "build_corridor_candidate_crops_for_town",
    "build_roundabout_crop_for_town",
    "load_scenarios",
    "solve_assignment",
    "plt",
    "mpatches",
]


if __name__ == "__main__":
    main()
