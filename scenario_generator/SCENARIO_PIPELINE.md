# Scenario Generator Pipeline

Concise map of the maintained scenario pipeline and where code should live.

## Canonical Entrypoint
- `scenario_generator/start_pipeline.py`
  - Main debug/production launcher for category-based generation.
  - Supports staged stops, multi-seed/category runs, target-mode batching, deferred CARLA validation.
- `tools/run_pipeline_debug.py`
  - Compatibility shim to the canonical entrypoint.

## Other Active Entrypoints
- `tools/run_audit_benchmark.py`
  - Benchmark harness with retry/repair loops and optional CARLA/leaderboard evaluation.
- `scenario_generator/run_scenario_pipeline.py`
  - Legacy free-text runner kept for compatibility.

## Pipeline Stages
1. `schema`: structured scenario generation.
2. `geometry`: derive geometry/topology constraints.
3. `crop`: crop-region selection.
4. `legal_paths`: legal path enumeration.
5. `pick_paths`: ego-path assignment.
6. `refine_paths`: spawn/speed/lane-change refinement.
7. `placement`: non-ego actor placement.
8. `validation`: deterministic rule checks.
9. `routes`: CARLA route XML + manifest emission.
10. `carla_validation`: final CARLA checks/repairs.

## Core Code Layout
- `scenario_generator/start_pipeline.py`
- `scenario_generator/carla_validation.py`
- `scenario_generator/schema_dashboard.py`
- `scenario_generator/scenario_generator/`
  - `capabilities.py`, `constraints.py`, `schema_generator.py`, `schema_utils.py`,
    `scene_validator.py`, `pipeline_runner.py`, `schema_generation_loop.py`
- `scenario_generator/pipeline/`
  - `step_01_crop` ... `step_07_route_alignment`

## Data Assets (kept)
- `scenario_generator/town_nodes/Town*.json`
- `scenario_generator/carla_assets.json`
- `scenario_generator/tests/artifacts/*` (test fixtures)

## Compatibility Policy
- Keep `tools/run_pipeline_debug.py`, `tools/carla_validation.py`, `tools/schema_dashboard.py` as thin shims.
- Keep wrappers (`run_crop_region_picker.py`, `run_path_picker.py`, `run_path_refiner.py`, `run_object_placer.py`, `convert_scene_to_routes.py`) because core modules import them.

## Cleanup Rules
- Do not check in generated run artifacts (`debug_runs/`, `scenario_generator/log/`, etc.).
- Remove ad-hoc one-off debug scripts unless they are wired into CI or called by active tooling.
- Prefer editing under `scenario_generator/` and leave `tools/` wrappers minimal.
