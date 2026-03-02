import time
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .llm_utils import _extract_first_json_object
from .models import GeometrySpec


class LLMGeometryExtractor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._mdl = None

    def _load(self):
        if self._tok is not None and self._mdl is not None:
            return
        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._mdl = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self._mdl.to(self.device)
        self._mdl.eval()

    def extract(self, scenario_text: str) -> GeometrySpec:
        self._load()

        schema = '''
Return JSON only, matching this schema exactly:
{
  "topology": "intersection" | "t_junction" | "corridor" | "unknown",
  "degree": 0 | 3 | 4,
  "required_maneuvers": {"straight": 0-3, "left": 0-3, "right": 0-3},
  "needs_oncoming": true|false,
  "needs_merge_onto_same_road": true|false,
  "needs_on_ramp": true|false,
  "needs_multi_lane": true|false,
  "min_lane_count": 1-3,
  "min_entry_runup_m": number,
  "min_exit_runout_m": number,
  "preferred_entry_cardinals": ["N","S","E","W"] or [],
  "avoid_extra_intersections": true|false,
  "confidence": number,
  "notes": "short"
}
'''
        prompt = (
            "You are a geometry spec extractor for CARLA driving scenarios.\n"
            "Read the scenario description and output the JSON schema below.\n"
            "Do not include any extra keys.\n"
            "If unsure, set fields to conservative defaults and lower confidence.\n"
            "If the topology is not stated, infer it from actions:\n"
            "- turns, perpendicular roads, oncoming traffic often imply intersections.\n"
            "- lane changes on a straight road imply corridor.\n"
            "- T-junction if a side road terminates into a main road.\n"
            "- an on-ramp merge implies needs_on_ramp=true and needs_merge_onto_same_road=true.\n"
            "\n"
            + schema
            + "\nScenario:\n"
            + scenario_text.strip()
            + "\nJSON:"
        )

        t0 = time.time()
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        out = self._mdl.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
            top_p=0.95,
            pad_token_id=self._tok.eos_token_id,
        )
        txt = self._tok.decode(out[0], skip_special_tokens=True)
        _ = time.time() - t0

        obj = _extract_first_json_object(txt)
        if obj is None:
            return self._fallback_spec(scenario_text, notes="parse_failed")

        try:
            return GeometrySpec(
                topology=str(obj.get("topology", "unknown")),
                degree=int(obj.get("degree", 0)),
                required_maneuvers=dict(obj.get("required_maneuvers", {"straight": 1, "left": 0, "right": 0})),
                needs_oncoming=bool(obj.get("needs_oncoming", False)),
                needs_merge_onto_same_road=bool(obj.get("needs_merge_onto_same_road", False)),
                needs_on_ramp=bool(obj.get("needs_on_ramp", False)),
                needs_multi_lane=bool(obj.get("needs_multi_lane", False)),
                min_lane_count=int(obj.get("min_lane_count", 1)),
                min_entry_runup_m=float(obj.get("min_entry_runup_m", 28.0)),
                min_exit_runout_m=float(obj.get("min_exit_runout_m", 18.0)),
                preferred_entry_cardinals=list(obj.get("preferred_entry_cardinals", [])) or [],
                avoid_extra_intersections=bool(obj.get("avoid_extra_intersections", True)),
                confidence=float(obj.get("confidence", 0.5)),
                notes=str(obj.get("notes", "")),
            )
        except Exception:
            return self._fallback_spec(scenario_text, notes="bad_fields")

    def _fallback_spec(self, scenario_text: str, notes: str = "") -> GeometrySpec:
        d = scenario_text.lower()
        topology = "intersection" if ("turn" in d or "junction" in d or "intersection" in d) else "corridor"
        if "t junction" in d or "t-junction" in d:
            topology = "t_junction"
        needs_ml = (
            ("change lanes" in d)
            or ("changes lanes" in d)
            or ("other lane" in d)
            or ("left lane" in d)
            or ("right lane" in d)
            or ("adjacent lane" in d)
            or ("multi-lane" in d)
            or ("two-lane" in d)
        )
        needs_oncoming = ("oncoming" in d) or ("opposite direction" in d)
        cardinals = set(re.findall(r"\b(north|south|east|west)\b", d))
        if ("north" in cardinals and "south" in cardinals) or ("east" in cardinals and "west" in cardinals):
            needs_oncoming = True
        needs_on_ramp = (
            ("on-ramp" in d)
            or ("on ramp" in d)
            or ("off-ramp" in d)
            or ("off ramp" in d)
            or ("ramp" in d and "merge" in d)
        )

        req = {"straight": 1, "left": 0, "right": 0}
        if "turn left" in d or "turns left" in d:
            req["left"] = 1
        if "turn right" in d or "turns right" in d:
            req["right"] = 1

        return GeometrySpec(
            topology=topology,
            degree=3 if topology == "t_junction" else 0,
            required_maneuvers=req,
            needs_oncoming=needs_oncoming,
            needs_merge_onto_same_road=(
                ("onto the road vehicle" in d and "traveling on" in d) or needs_on_ramp
            ),
            needs_on_ramp=needs_on_ramp,
            needs_multi_lane=needs_ml,
            min_lane_count=2 if needs_ml else 1,
            min_entry_runup_m=40.0 if ("spawns behind" in d or "chain" in d) else 28.0,
            min_exit_runout_m=28.0 if ("after exiting" in d or "after turning" in d) else 18.0,
            preferred_entry_cardinals=[],
            avoid_extra_intersections=True,
            confidence=0.15,
            notes=f"fallback:{notes}",
        )
