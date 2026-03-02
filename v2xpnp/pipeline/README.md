# V2XPNP Pipeline Layout

This package now contains the canonical trajectory pipeline implementation.

## Modules
- `pipeline_runtime.py`
  - Canonical runtime namespace assembled from stage internals.
- `runtime_common.py`
  - Shared types, env/config helpers, map loading, lane matching/alignment classes.
- `runtime_projection.py`
  - CARLA projection and projection-time smoothing/guards.
- `runtime_postprocess.py`
  - Overlap reduction and actor dedup postprocess.
- `runtime_orchestration.py`
  - Dataset assembly, HTML generation, and runtime CLI orchestration.
- `route_export.py`
  - Canonical route export namespace assembled from stage internals.
- `route_export_stage_01_foundation.py` .. `route_export_stage_04_orchestration.py`
  - Route export internals split by lifecycle stage.
- `trajectory_ingest.py`
  - Canonical trajectory ingest namespace assembled from stage internals.
- `trajectory_ingest_stage_01_types_io.py` .. `trajectory_ingest_stage_04_cli.py`
  - Ingest internals split by lifecycle stage.
- `correspondence_diagnostics.py`
  - Lane correspondence diagnostics CLI and analysis module.
- `entrypoint.py`
  - Stable CLI entrypoint that delegates to `pipeline_runtime.main()`.
- `stages.py` + `stage_00_cli_config.py` .. `stage_05_output_rendering.py`
  - High-level stage facade modules with explicit step boundaries.
- `audit.py`
  - Post-run audit utility for optimizer outputs, with emphasis on wrong-way/catastrophic/intersection blockers.

## Canonical CLI Commands
- Main map plotting + projection pipeline:
  - `python3 -m v2xpnp.pipeline.entrypoint <scenario_dir>`
- Route export:
  - `python3 -m v2xpnp.pipeline.route_export ...`
- Trajectory ingest to CARLA logs:
  - `python3 -m v2xpnp.pipeline.trajectory_ingest ...`
- Lane correspondence diagnostics:
  - `python3 -m v2xpnp.pipeline.correspondence_diagnostics ...`
