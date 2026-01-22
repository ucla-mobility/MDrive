"""
Schema-driven generation loop.

Generates JSON scenario specs, runs the pipeline, and validates outputs.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .capabilities import CATEGORY_DEFINITIONS, TopologyType, get_available_categories
from .constraints import spec_to_dict
from .pipeline_runner import GenerationLogger, PipelineRunner
from .scene_validator import SceneValidator, SceneValidationResult
from .schema_generator import SchemaGenerationConfig, SchemaScenarioGenerator
from .schema_utils import geometry_spec_from_scenario_spec


DEFAULT_SCHEMA_CATEGORIES = [
    "Unprotected Left Turn",
    "Interactive Lane Change",
    "Major/Minor Unsignalized Entry",
]


@dataclass
class SchemaGenerationResult:
    scenario_id: str
    category: str
    description: str
    success: bool
    scene_path: Optional[str] = None
    validation_result: Optional[SceneValidationResult] = None
    attempts: int = 0
    error_message: Optional[str] = None
    spec: Optional[dict] = None


@dataclass
class SchemaGenerationLoopConfig:
    categories: Optional[List[str]] = None
    variants_per_category: int = 1
    output_dir: str = "log_schema"
    town: str = "Town05"
    highway_town: Optional[str] = "Town06"
    t_junction_town: Optional[str] = "Town02"
    two_lane_corridor_town: Optional[str] = "Town02"
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    viz_objects: bool = True
    routes_out_dir: str = "routes"
    routes_ego_num: Optional[int] = None
    align_routes: bool = True
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    max_retries_per_scenario: int = 3
    min_validation_score: float = 0.3
    mute_pipeline: bool = True
    template_only: bool = False
    verbose: bool = False
    allow_template_fallback: bool = False


class SchemaGenerationLoop:
    def __init__(self, config: SchemaGenerationLoopConfig, model=None, tokenizer=None):
        self.config = config
        self.logger = GenerationLogger(verbose=config.verbose)

        if model is None or tokenizer is None:
            self.logger.info(f"Loading model: {config.model_id}")
            model, tokenizer = self._load_model(config.model_id)

        self._model = model
        self._tokenizer = tokenizer

        schema_config = SchemaGenerationConfig(
            model_id=config.model_id,
            allow_template_fallback=config.allow_template_fallback,
        )
        self.generator = SchemaScenarioGenerator(
            schema_config,
            model=model,
            tokenizer=tokenizer,
            template_only=config.template_only,
        )
        self.scene_validator = SceneValidator()
        self.pipeline_runner = PipelineRunner(
            config.output_dir,
            model=model,
            tokenizer=tokenizer,
            viz_objects=config.viz_objects,
            routes_out_dir=config.routes_out_dir,
            routes_ego_num=config.routes_ego_num,
        )

        self.categories = config.categories or DEFAULT_SCHEMA_CATEGORIES
        self.categories = [c for c in self.categories if c in get_available_categories()]

        self.generated_results: List[SchemaGenerationResult] = []
        self.failed_results: List[SchemaGenerationResult] = []

        self.total_attempts = 0
        self.total_pipeline_runs = 0
        self.total_validation_failures = 0

    def _load_model(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        return model, tokenizer

    def _safe_name(self, text: str) -> str:
        return text.replace(" ", "_").replace("/", "_")

    def _resolve_town(self, category: str) -> str:
        info = CATEGORY_DEFINITIONS.get(category)
        if info:
            if self.config.highway_town and info.map.topology == TopologyType.HIGHWAY:
                return self.config.highway_town
            if self.config.t_junction_town and info.map.topology == TopologyType.T_JUNCTION:
                return self.config.t_junction_town
            if self.config.two_lane_corridor_town and info.map.topology == TopologyType.TWO_LANE_CORRIDOR:
                return self.config.two_lane_corridor_town
        return self.config.town

    def _write_spec_files(self, out_dir: Path, spec: dict, description: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = out_dir / "scenario_spec.json"
        desc_path = out_dir / "scenario_description.txt"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        desc_path.write_text(description, encoding="utf-8")

    def _log_pipeline_outputs(self, out_dir: Path) -> None:
        if not self.config.verbose:
            return
        expected = [
            "scenario_input.txt",
            "legal_paths_prompt.txt",
            "legal_paths_detailed.json",
            "path_picker_prompt.txt",
            "picked_paths_detailed.json",
            "picked_paths_refined.json",
            "scene_objects.json",
            "scene_objects.png",
        ]
        present = [name for name in expected if (out_dir / name).exists()]
        missing = [name for name in expected if not (out_dir / name).exists()]
        if present:
            self.logger.info(f"Pipeline outputs: {', '.join(present)}")
        if missing:
            self.logger.warning(f"Missing outputs: {', '.join(missing)}")

    def _routes_dir_for(self, category: str, variant_index: int) -> Path:
        clean_id = self._safe_name(category)
        if variant_index > 1:
            clean_id = f"{clean_id}_v{variant_index}"
        return Path(self.config.routes_out_dir) / clean_id

    def generate_single(self, category: str, variant_index: int) -> SchemaGenerationResult:
        base_id = self._safe_name(category)
        if variant_index > 1:
            base_id = f"{base_id}_v{variant_index}"

        last_error = None
        for attempt in range(1, self.config.max_retries_per_scenario + 1):
            self.total_attempts += 1
            scenario_id = f"{base_id}_attempt{attempt}"

            spec_obj, errors, _warnings = self.generator.generate_spec(category)
            if spec_obj is None:
                last_error = "; ".join(errors) if errors else "spec_generation_failed"
                continue

            geometry_spec = geometry_spec_from_scenario_spec(spec_obj)
            spec_dict = spec_to_dict(spec_obj)
            schema_text = json.dumps(spec_dict, indent=2, sort_keys=True)
            town = self._resolve_town(category)

            out_dir = Path(self.config.output_dir) / self._safe_name(scenario_id)
            self._write_spec_files(out_dir, spec_dict, schema_text)
            self.logger.info(f"Spec outputs in: {out_dir.resolve()}")
            self.logger.info("Pipeline steps: crop -> legal_paths -> path_picker -> path_refiner -> object_placer")
            self.logger.info(f"Town: {town}")

            self.total_pipeline_runs += 1
            success, scene_path, error_msg = self.pipeline_runner.run_full_pipeline(
                scenario_text=schema_text,
                scenario_schema=spec_dict,
                category=category,
                scenario_id=scenario_id,
                town=town,
                mute=self.config.mute_pipeline,
                model_id=self.config.model_id,
                geometry_spec=geometry_spec,
            )
            if not success:
                last_error = error_msg or "pipeline_failed"
                self.logger.warning(f"Pipeline failed; outputs (if any) in: {out_dir.resolve()}")
                continue
            if scene_path:
                self.logger.info(f"Scene output: {scene_path}")
            self._log_pipeline_outputs(out_dir)

            validation = self.scene_validator.validate_scene(
                scene_path,
                schema_text,
                category=category,
                scenario_spec=spec_dict,
            )
            if validation.score < self.config.min_validation_score:
                self.total_validation_failures += 1
                last_error = f"validation_score={validation.score:.2f}"
                continue

            # Successful scenario
            if self.config.routes_out_dir and scene_path:
                try:
                    from convert_scene_to_routes import convert_scene_to_routes

                    routes_dir = self._routes_dir_for(category, variant_index)
                    convert_scene_to_routes(
                        str(scene_path),
                        str(routes_dir),
                        ego_num=self.config.routes_ego_num,
                        align_routes=self.config.align_routes,
                        carla_host=self.config.carla_host,
                        carla_port=self.config.carla_port,
                    )
                except Exception as exc:
                    self.logger.warning(f"Routes generation failed for {scenario_id}: {exc}")

            return SchemaGenerationResult(
                scenario_id=scenario_id,
                category=category,
                description=schema_text,
                success=True,
                scene_path=scene_path,
                validation_result=validation,
                attempts=attempt,
                spec=spec_dict,
            )

        return SchemaGenerationResult(
            scenario_id=base_id,
            category=category,
            description="",
            success=False,
            attempts=self.config.max_retries_per_scenario,
            error_message=last_error,
        )

    def run(self) -> List[SchemaGenerationResult]:
        self.logger.divider()
        self.logger.info("Starting schema generation loop")
        self.logger.info(f"Categories: {len(self.categories)}, Variants per category: {self.config.variants_per_category}")
        output_root = Path(self.config.output_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Schema output dir: {output_root}")
        self.logger.divider()

        start_time = time.time()

        for category in self.categories:
            for idx in range(1, self.config.variants_per_category + 1):
                self.logger.info(f"Generating {category} (variant {idx})")
                result = self.generate_single(category, idx)
                if result.success:
                    self.generated_results.append(result)
                    score = result.validation_result.score if result.validation_result else 0.0
                    self.logger.success(
                        f"Generated {category} variant {idx} (attempts={result.attempts}, score={score:.2f})"
                    )
                else:
                    self.failed_results.append(result)
                    self.logger.warning(
                        f"Failed {category} variant {idx} after {result.attempts} attempts: {result.error_message}"
                    )

        elapsed = time.time() - start_time
        self.logger.divider()
        self.logger.info("Schema generation complete")
        self.logger.info(f"Successful: {len(self.generated_results)}")
        self.logger.info(f"Failed: {len(self.failed_results)}")
        self.logger.info(f"Attempts: {self.total_attempts}")
        self.logger.info(f"Pipeline runs: {self.total_pipeline_runs}")
        self.logger.info(f"Validation failures: {self.total_validation_failures}")
        self.logger.info(f"Elapsed: {elapsed:.1f}s")
        self.logger.divider()

        self._save_results()
        return self.generated_results

    def _save_results(self) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        scenarios_path = out_dir / "schema_scenarios.json"
        results_path = out_dir / "schema_generation_results.json"
        log_path = out_dir / "schema_generation_log.txt"

        scenarios = {}
        for result in self.generated_results:
            scenarios.setdefault(result.category, []).append(
                {
                    "spec": result.spec,
                    "description": result.description,
                    "scenario_id": result.scenario_id,
                    "scene_path": result.scene_path,
                }
            )

        results = {
            "generated": [
                {
                    "scenario_id": r.scenario_id,
                    "category": r.category,
                    "success": r.success,
                    "attempts": r.attempts,
                    "scene_path": r.scene_path,
                    "error_message": r.error_message,
                }
                for r in self.generated_results
            ],
            "failed": [
                {
                    "scenario_id": r.scenario_id,
                    "category": r.category,
                    "success": r.success,
                    "attempts": r.attempts,
                    "error_message": r.error_message,
                }
                for r in self.failed_results
            ],
        }

        scenarios_path.write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        self.logger.save(str(log_path))
