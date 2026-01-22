"""
Pipeline runner wrapper for schema-based generation.

Provides:
- GenerationLogger for consistent timestamps.
- mute_stdout to suppress noisy pipeline output.
- PipelineRunner to execute the scenario pipeline directly.
"""

import contextlib
import io
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


class GenerationLogger:
    """Clean logging for generation loops, with elapsed timestamps."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.log_buffer = []
        self.start_time = time.time()

    def _timestamp(self) -> str:
        elapsed = time.time() - self.start_time
        return f"[{elapsed:6.1f}s]"

    def info(self, msg: str):
        line = f"{self._timestamp()} [INFO] {msg}"
        print(line)
        self.log_buffer.append(line)

    def success(self, msg: str):
        line = f"{self._timestamp()} [OK] {msg}"
        print(line)
        self.log_buffer.append(line)

    def warning(self, msg: str):
        line = f"{self._timestamp()} [WARN] {msg}"
        print(line)
        self.log_buffer.append(line)

    def divider(self):
        print("-" * 60)

    def save(self, path: str):
        with open(path, "w") as f:
            f.write("\n".join(self.log_buffer))


@contextlib.contextmanager
def mute_stdout():
    """Context manager to mute stdout/stderr (for suppressing pipeline output)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _schema_to_text(schema: Optional[Dict[str, Any]]) -> str:
    """Pretty-print a schema dict for prompts/logging."""
    if not schema:
        return ""
    try:
        return json.dumps(schema, indent=2, sort_keys=True)
    except Exception:
        return str(schema)


def _constraints_from_schema(schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Build a path-picker constraints payload directly from a schema dict.

    This mirrors the structure expected by step_03_path_picker:
    {
      "vehicles": [{"vehicle": "Vehicle 1", "maneuver": "left", ...}],
      "constraints": [{"type": "same_approach_as", "a": "Vehicle 1", "b": "Vehicle 2"}]
    }
    """
    if not schema or not isinstance(schema, dict):
        return None

    vehicles = []
    for ego in schema.get("ego_vehicles", []):
        if not isinstance(ego, dict):
            continue
        vid = ego.get("vehicle_id")
        if not vid:
            continue
        vehicles.append(
            {
                "vehicle": str(vid),
                "maneuver": str(ego.get("maneuver", "unknown")).lower(),
                "lane_change_phase": str(ego.get("lane_change_phase", "unknown")).lower(),
                "approach_direction": "unknown",
                "entry_road": str(ego.get("entry_road", "unknown")).lower(),
                "exit_road": str(ego.get("exit_road", "unknown")).lower(),
            }
        )

    constraints = []
    for c in schema.get("vehicle_constraints", []):
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        a = c.get("a")
        b = c.get("b")
        if not ctype or not a or not b:
            continue
        constraints.append(
            {
                "type": str(ctype).lower(),
                "a": str(a),
                "b": str(b),
            }
        )

    if not vehicles:
        return None

    return {"vehicles": vehicles, "constraints": constraints}


def _normalize_lateral_for_stage1(value: Optional[str]) -> Optional[str]:
    """Clamp lateral positions to the small set allowed by stage1 prompts."""
    allowed = {
        "center",
        "half_right",
        "right_edge",
        "right_lane",
        "half_left",
        "left_edge",
        "left_lane",
    }
    if not value:
        return None
    val = str(value).lower()
    if val in allowed:
        return val
    # Map offroad/sidewalk variants to edges to avoid rejection
    if "right" in val:
        return "right_edge"
    if "left" in val:
        return "left_edge"
    return None


def _entities_from_schema(schema: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Build Stage 1 object-placer entities directly from schema actors.

    This bypasses the need to re-extract actors from natural language.
    """
    if not schema or not isinstance(schema, dict):
        return None

    actors = schema.get("actors", [])
    if not isinstance(actors, list) or not actors:
        return []

    entities: List[Dict[str, Any]] = []
    for idx, actor in enumerate(actors):
        if not isinstance(actor, dict):
            continue
        ent_id = f"entity_{idx + 1}"
        mention = str(actor.get("actor_id", ent_id))
        evidence = mention  # evidence must be substring of description; schema text will include actor_id
        motion = str(actor.get("motion", "unknown")).lower()
        if motion == "cross_perpendicular":
            motion_hint = "crossing"
        elif motion == "follow_lane":
            motion_hint = "follow_lane"
        elif motion == "static":
            motion_hint = "static"
        else:
            motion_hint = "unknown"

        direction_rel = None
        dir_rel_val = actor.get("direction_relative_to")
        if dir_rel_val and actor.get("affects_vehicle"):
            direction_rel = {"vehicle": actor.get("affects_vehicle"), "direction": dir_rel_val}

        entities.append(
            {
                "entity_id": ent_id,
                "mention": mention,
                "evidence": evidence,
                "actor_kind": actor.get("kind", "static_prop"),
                "quantity": actor.get("quantity", 1),
                "group_pattern": actor.get("group_pattern", "unknown"),
                "start_lateral": _normalize_lateral_for_stage1(actor.get("start_lateral")),
                "end_lateral": _normalize_lateral_for_stage1(actor.get("end_lateral")),
                "affects_vehicle": actor.get("affects_vehicle"),
                "when": actor.get("timing_phase", "unknown"),
                "lateral_relation": _normalize_lateral_for_stage1(actor.get("lateral_position")) or "unknown",
                "motion_hint": motion_hint,
                "crossing_direction": actor.get("crossing_direction"),
                "speed_hint": actor.get("speed", "unknown"),
                "trigger": None,
                "action": None,
                "direction_relative_to": direction_rel,
            }
        )

    return entities


class PipelineRunner:
    """
    Wrapper around the scenario pipeline with muted output and structured results.

    Runs the pipeline in-process using shared model/tokenizer.
    """

    def __init__(
        self,
        output_base_dir: str = "log",
        model=None,
        tokenizer=None,
        viz_objects: bool = True,
        routes_out_dir: str = "routes",
        routes_ego_num: Optional[int] = None,
        refiner_max_new_tokens: int = 2048,
        placer_max_new_tokens: int = 2048,
        placer_do_sample: bool = False,
        placer_temperature: float = 0.2,
        placer_top_p: float = 0.95,
    ):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.viz_objects = viz_objects
        self.routes_out_dir = routes_out_dir
        self.routes_ego_num = routes_ego_num
        self.refiner_max_new_tokens = refiner_max_new_tokens
        self.placer_max_new_tokens = placer_max_new_tokens
        self.placer_do_sample = placer_do_sample
        self.placer_temperature = placer_temperature
        self.placer_top_p = placer_top_p

        self._model = model
        self._tokenizer = tokenizer
        self._model_loaded = model is not None

    def _ensure_model(self, model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"):
        """Load the shared model if not already loaded."""
        if self._model_loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()
        self._model_loaded = True

    def run_full_pipeline(
        self,
        category: str,
        scenario_id: str,
        scenario_text: str = "",
        scenario_schema: Optional[Dict[str, Any]] = None,
        town: str = "Town05",
        mute: bool = True,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
        geometry_spec: Optional[Any] = None,
        stats: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run the full pipeline for a scenario using the shared model.

        Returns:
            (success, scene_objects_path, error_message)
        """
        self._ensure_model(model_id)

        schema_text = _schema_to_text(scenario_schema)
        payload_text = scenario_text or schema_text

        safe_name = scenario_id.replace(" ", "_").replace("/", "_")
        out_dir = self.output_base_dir / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "scenario_input.txt", "w") as f:
            f.write(f"Category: {category}\n")
            f.write(f"Town: {town}\n")
            if schema_text:
                f.write("Schema:\n")
                f.write(schema_text + "\n")
            if payload_text and payload_text != schema_text:
                f.write("Payload text:\n")
                f.write(payload_text + "\n")

        try:
            if mute:
                with mute_stdout():
                    return self._run_pipeline_direct(
                        out_dir,
                        payload_text,
                        town,
                        scenario_id,
                        category=category,
                        scenario_schema=scenario_schema,
                        geometry_spec=geometry_spec,
                        stats=stats,
                    )
            return self._run_pipeline_direct(
                out_dir,
                payload_text,
                town,
                scenario_id,
                category=category,
                scenario_schema=scenario_schema,
                geometry_spec=geometry_spec,
                stats=stats,
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            return False, None, error_msg

    def _run_pipeline_direct(
        self,
        out_dir: Path,
        scenario_text: str,
        town: str,
        scenario_id: str,
        category: str = "",
        scenario_schema: Optional[Dict[str, Any]] = None,
        geometry_spec: Optional[Any] = None,
        stats: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Run the scenario pipeline directly using the shared model."""
        from .capabilities import CATEGORY_DEFINITIONS, TopologyType

        require_straight = False
        cat_info = CATEGORY_DEFINITIONS.get(category)
        if cat_info and cat_info.map.topology in {TopologyType.CORRIDOR, TopologyType.TWO_LANE_CORRIDOR, TopologyType.HIGHWAY}:
            require_straight = True
            print(f"[INFO] Category '{category}' requires {cat_info.map.topology.value} topology, filtering to straight paths")

        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from generate_legal_paths import (
            CropBox,
            build_connectivity,
            build_path_signature,
            build_segments,
            crop_segments,
            generate_legal_paths,
            load_nodes,
            make_path_name,
            save_aggregated_signatures_json,
            save_prompt_file,
            build_segments_detailed_for_path,
        )
        from run_path_picker import pick_paths_with_model
        from run_path_refiner import refine_picked_paths_with_model
        from run_object_placer import run_object_placer
        import run_crop_region_picker as crop_picker
        from pipeline.step_01_crop.scoring import crop_satisfies_spec

        nodes_path = parent_dir / "town_nodes" / f"{town}.json"
        if not nodes_path.exists():
            return False, None, f"Nodes file not found: {nodes_path}"

        legal_prompt_path = out_dir / "legal_paths_prompt.txt"
        legal_json_path = out_dir / "legal_paths_detailed.json"
        picker_prompt_path = out_dir / "path_picker_prompt.txt"
        picked_paths_path = out_dir / "picked_paths_detailed.json"
        refined_paths_path = out_dir / "picked_paths_refined.json"
        scene_json_path = out_dir / "scene_objects.json"
        scene_png_path = out_dir / "scene_objects.png"

        schema_constraints = _constraints_from_schema(scenario_schema)
        schema_entities = _entities_from_schema(scenario_schema)

        if geometry_spec is None:
            extractor = crop_picker.LLMGeometryExtractor(model_name="", device="cuda")
            extractor._tok = self._tokenizer
            extractor._mdl = self._model
            spec = extractor.extract(scenario_text)
        else:
            spec = geometry_spec

        radii = [45.0, 55.0, 65.0]
        crops = crop_picker.build_candidate_crops_for_town(
            town_name=town,
            town_json_path=str(nodes_path),
            radii=radii,
            min_path_len=22.0,
            max_paths=80,
            max_depth=8,
        )

        if not crops:
            return False, None, f"No candidate crops found for {town}"

        count_on_ramp = sum(1 for c in crops if getattr(c, "has_on_ramp", False))
        count_merge = sum(1 for c in crops if getattr(c, "has_merge_onto_same_road", False))
        count_multi_lane = sum(1 for c in crops if getattr(c, "has_multi_lane", False))
        count_oncoming = sum(1 for c in crops if getattr(c, "has_oncoming_pair", False))

        if spec.needs_on_ramp and count_on_ramp == 0:
            return False, None, (
                f"No crops with on-ramp geometry found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )
        if spec.needs_merge_onto_same_road and count_merge == 0:
            return False, None, (
                f"No crops with merge geometry found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )
        if spec.needs_multi_lane and count_multi_lane == 0:
            return False, None, (
                f"No multi-lane crops found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )
        if spec.needs_oncoming and count_oncoming == 0:
            return False, None, (
                f"No oncoming-pair crops found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )

        scenario = crop_picker.Scenario(sid=scenario_id, text=scenario_text)
        res = crop_picker.solve_assignment(
            scenarios=[scenario],
            specs={scenario_id: spec},
            crops=crops,
            domain_k=50,
            capacity_per_crop=10,
            reuse_weight=4000.0,
            junction_penalty=25000.0,
            log_every=0,
        )

        assignments = res.detailed.get("assignments", {})
        if scenario_id not in assignments:
            satisfying = [c for c in crops if crop_satisfies_spec(spec, c)]
            if not satisfying:
                return False, None, (
                    f"No crops satisfy geometry spec; "
                    f"needs_on_ramp={spec.needs_on_ramp}, "
                    f"needs_merge_onto_same_road={spec.needs_merge_onto_same_road}, "
                    f"needs_multi_lane={spec.needs_multi_lane}, "
                    f"needs_oncoming={spec.needs_oncoming}"
                )
            cand = min(satisfying, key=lambda c: c.area)
            crop_vals = [cand.crop.xmin, cand.crop.xmax, cand.crop.ymin, cand.crop.ymax]
        else:
            crop_vals = assignments[scenario_id]["crop"]
        crop = CropBox(xmin=crop_vals[0], xmax=crop_vals[1], ymin=crop_vals[2], ymax=crop_vals[3])

        data = load_nodes(str(nodes_path))
        all_segments = build_segments(data)
        cropped_segments = crop_segments(all_segments, crop)
        if not cropped_segments:
            return False, None, "No segments found in crop region"

        adj = build_connectivity(
            cropped_segments,
            connect_radius_m=6.0,
            connect_yaw_tol_deg=60.0,
        )
        max_paths = 200 if spec.needs_on_ramp else 100
        max_depth = 8 if spec.needs_on_ramp else 5
        legal_paths = generate_legal_paths(
            cropped_segments,
            adj,
            crop,
            min_path_length=20.0,
            max_paths=max_paths,
            max_depth=max_depth,
            allow_within_region_fallback=False,
        )
        if not legal_paths:
            return False, None, "No legal paths found for this crop"

        params = {
            "max_yaw_diff_deg": 60.0,
            "connect_radius_m": 6.0,
            "min_path_length_m": 20.0,
            "max_paths": max_paths,
            "max_depth": max_depth,
            "turn_frame": "WORLD_FRAME",
        }

        candidates = []
        for i, path in enumerate(legal_paths):
            sig = build_path_signature(path)
            name = make_path_name(i, sig)
            sig["segments_detailed"] = build_segments_detailed_for_path(path, polyline_sample_n=10)
            candidates.append({"name": name, "signature": sig})

        lane_ids_by_road: Dict[int, set] = {}
        for s in cropped_segments:
            try:
                rid = int(s.road_id)
                lid = int(s.lane_id)
            except Exception:
                continue
            lane_ids_by_road.setdefault(rid, set()).add(lid)
        lane_counts_by_road = {rid: len(lanes) for rid, lanes in lane_ids_by_road.items()}

        save_prompt_file(
            str(legal_prompt_path),
            crop=crop,
            nodes_path=str(nodes_path),
            params=params,
            paths_named=candidates,
        )
        save_aggregated_signatures_json(
            str(legal_json_path),
            crop=crop,
            nodes_path=str(nodes_path),
            params=params,
            paths_named=candidates,
            lane_counts_by_road=lane_counts_by_road,
        )

        prompt_text = legal_prompt_path.read_text(encoding="utf-8").strip()
        prompt_text += (
            "\n\nUSER SCENARIO SCHEMA (JSON):\n"
            + scenario_text
            + "\nUse this structured schema (not prose) to choose ego paths and constraints.\n"
            + "(Only assign paths to moving vehicles; ignore static/parked props.)\n"
        )
        picker_prompt_path.write_text(prompt_text, encoding="utf-8")

        require_on_ramp = bool(getattr(spec, "needs_on_ramp", False))
        # Use sampling with per-scenario seed to encourage variation across repair attempts
        try:
            import torch
            seed_val = abs(hash(scenario_id)) % (2**31)
            torch.manual_seed(seed_val)
        except Exception:
            pass

        pick_paths_with_model(
            prompt=prompt_text,
            aggregated_json=str(legal_json_path),
            out_picked_json=str(picked_paths_path),
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.4,
            top_p=0.95,
            require_straight=require_straight,
            require_on_ramp=require_on_ramp,
            schema_constraints=schema_constraints,
        )

        refine_picked_paths_with_model(
            picked_paths_json=str(picked_paths_path),
            description=scenario_text,
            out_json=str(refined_paths_path),
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.refiner_max_new_tokens,
            carla_assets=str(parent_dir / "carla_assets.json"),
            schema_payload=scenario_schema,
        )

        from types import SimpleNamespace
        placer_args = SimpleNamespace(
            model="",
            picked_paths=str(refined_paths_path),
            carla_assets=str(parent_dir / "carla_assets.json"),
            description=scenario_text,
            out=str(scene_json_path),
            viz_out=str(scene_png_path),
            viz=self.viz_objects,
            viz_show=False,
            nodes_root=str(nodes_path.parent),
            max_new_tokens=self.placer_max_new_tokens,
            do_sample=self.placer_do_sample,
            temperature=self.placer_temperature,
            top_p=self.placer_top_p,
            placement_mode="csp",
            schema_entities=schema_entities,
        )
        if stats is not None:
            placer_args.stats = stats
        run_object_placer(placer_args, model=self._model, tokenizer=self._tokenizer)

        if scene_json_path.exists():
            return True, str(scene_json_path), None
        return False, None, "scene_objects.json not created"
