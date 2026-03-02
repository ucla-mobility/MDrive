# scenario_generator package

Core package for schema-first scenario generation and validation.

## Primary usage
- Main launcher: `python scenario_generator/start_pipeline.py --category "Construction Zone" --seed 42`
- Benchmark harness: `python tools/run_audit_benchmark.py ...`

## Important modules
- `capabilities.py`: category/topology requirements.
- `constraints.py`: `ScenarioSpec` and related schema types.
- `schema_generator.py`: LLM/template schema generation.
- `schema_utils.py`: schema-to-geometry transforms.
- `scene_validator.py`: deterministic scene checks.
- `pipeline_runner.py`: orchestrates crop/path/placement pipeline calls.
- `schema_generation_loop.py`: repeated schema generation + pipeline execution loops.

## Notes
- Wrapper scripts in `scenario_generator/` and `tools/` are retained for compatibility.
- Keep new pipeline features in `scenario_generator/start_pipeline.py` and shared modules here.
