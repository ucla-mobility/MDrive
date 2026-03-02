import json
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract any top-level JSON object from arbitrary text using a balanced-brace scan.
    """
    # First try whole text
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start_search = 0
    while True:
        start = text.find("{", start_search)
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = text[start : i + 1]
                        try:
                            obj = json.loads(snippet)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
        start_search = start + 1


def _chat_template(tokenizer: Any) -> bool:
    return callable(getattr(tokenizer, "apply_chat_template", None))


def _llm_generate_json(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    do_sample: bool = False,
) -> Optional[Dict[str, Any]]:
    if _chat_template(tokenizer):
        messages = [
            {"role": "system", "content": "You are a careful constraint extractor. You only output JSON."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if torch and torch.cuda.is_available():
            input_ids = input_ids.to(model.device)
        attn = (input_ids != tokenizer.pad_token_id).long()
        gen_kwargs = {"input_ids": input_ids, "attention_mask": attn}
        input_len = int(input_ids.shape[-1])
    else:
        enc = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch and torch.cuda.is_available():
            enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs = enc
        input_len = int(enc["input_ids"].shape[-1])

    # Build generation kwargs; omit temperature/top_p when not sampling to avoid warnings
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_config["temperature"] = temperature
        gen_config["top_p"] = top_p

    print(f"[DEBUG] refiner LLM: prompt_tokens={input_len}, max_new={max_new_tokens}, do_sample={do_sample}", flush=True)
    with torch.no_grad():
        out = model.generate(**gen_kwargs, **gen_config)
    print(f"[DEBUG] refiner LLM: generation complete, output_tokens={out.shape[-1] - input_len}", flush=True)

    gen = out[0][input_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return _extract_first_json_object(text)


def extract_refinement_constraints(
    description: str,
    vehicles: List[str],
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    schema_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "vehicle_speeds": [{"vehicle":"Vehicle 1", "speed_class":"slow|normal|fast"}],
        "spawn_relations": [
           {"type":"ahead_of|behind_of", "a":"Vehicle X", "b":"Vehicle Y", "distance_m":10, "tolerance_m":5, "allow_other_lane":true}
        ],
        "lane_changes": [
           {"vehicle":"Vehicle X", "type":"merge_into_lane_of", "target":"Vehicle Y", "style":"cut_off|polite", "timing":"near_conflict|asap"}
        ],
        "options": {"synchronize_conflicts": true}
      }
    """
    # Intentional, narrow schema with explicit allowed values.
    vehicles_list = ", ".join(vehicles)
    prompt = f"""
You will read a driving scene description.

Your job: extract ONLY the minimal, explicitly-stated refinement requests that affect EGO vehicle spawn points,
and/or require inserting a lane-change into an ego path.

EGO vehicles are ONLY: {vehicles_list}

IMPORTANT:
- Only output fields you are confident are explicitly described.
- If a constraint is not explicitly described, OMIT it (do NOT guess).
- Do NOT invent new constraint types. Use ONLY the allowed schema below.
- If there are no refinements, return empty lists.

ALLOWED OUTPUT JSON SCHEMA (return JSON only):
{{
  "vehicle_speeds": [
    {{ "vehicle": "Vehicle 1", "speed_class": "slow" | "normal" | "fast" }}
  ],
  "spawn_relations": [
    {{
      "type": "ahead_of" | "behind_of",
      "a": "Vehicle X",
      "b": "Vehicle Y",
      "distance_m": <number, optional>,
      "tolerance_m": <number, optional>,
      "allow_other_lane": <true|false, optional>
    }}
  ],
  "lane_changes": [
    {{
      "vehicle": "Vehicle X",
      "type": "merge_into_lane_of",
      "target": "Vehicle Y",
      "style": "cut_off" | "polite",
      "timing": "near_conflict" | "asap",
      "phase": "before_intersection" | "in_intersection" | "after_intersection" | "unknown"
    }}
  ],
  "options": {{
    "synchronize_conflicts": <true|false, optional>
  }}
}}

Mapping guidance:
- If description says "slow" or "at a slow speed" -> speed_class="slow".
- If description says "fast" or "accelerates" -> speed_class="fast".
- Otherwise omit speed (we'll default to normal).
- "spawns in front of" -> spawn_relations: type="ahead_of".
- "spawns behind" -> spawn_relations: type="behind_of".
- "changes lanes into the same lane as Vehicle Y" -> lane_changes: merge_into_lane_of (style likely "cut_off" if "cut off").
- "changes lanes after the intersection" -> lane_changes: phase="after_intersection".
- "changes lanes before the intersection" -> lane_changes: phase="before_intersection".
- If lane change timing relative to intersection is not specified, use phase="unknown".

Scene description:
{description}
""".strip()

    if schema_payload is not None:
        # Structured schema provided: derive lane_changes from vehicle_constraints.
        lane_changes: List[Dict[str, Any]] = []
        for vc in schema_payload.get("vehicle_constraints", []) or []:
            if not isinstance(vc, dict):
                continue
            t = str(vc.get("type", "")).lower()
            if t not in ("merges_into_lane_of", "merge_into_lane_of"):
                continue
            a = vc.get("a") or vc.get("vehicle")
            b = vc.get("b") or vc.get("target")
            if a not in vehicles or b not in vehicles:
                continue
            lane_changes.append({
                "vehicle": a,
                "type": "merge_into_lane_of",
                "target": b,
                "style": "polite",
                "timing": "unknown",
                "phase": "unknown",
            })
        return {"vehicle_speeds": [], "spawn_relations": [], "lane_changes": lane_changes, "options": {"synchronize_conflicts": True}}

    if model is None or tokenizer is None:
        # No model provided: default to no extra constraints
        return {"vehicle_speeds": [], "spawn_relations": [], "lane_changes": [], "options": {"synchronize_conflicts": True}}

    desc_lc = description.lower()

    obj = _llm_generate_json(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    if not isinstance(obj, dict):
        return {"vehicle_speeds": [], "spawn_relations": [], "lane_changes": [], "options": {"synchronize_conflicts": True}}

    # Filter + sanitize: keep only allowed vehicles and allowed keys
    vset = set(vehicles)
    out = {"vehicle_speeds": [], "spawn_relations": [], "lane_changes": [], "options": {"synchronize_conflicts": True}}

    # options
    # Default behavior is to synchronize conflicts. We only allow the model to DISABLE
    # synchronization when the scene text explicitly asks for staggered / non-simultaneous
    # arrivals. This avoids the model "helpfully" outputting false and accidentally
    # turning off the optimization.
    opt = obj.get("options")
    if isinstance(opt, dict) and "synchronize_conflicts" in opt:
        requested = bool(opt.get("synchronize_conflicts"))
        if requested:
            out["options"]["synchronize_conflicts"] = True
        else:
            disable_triggers = [
                "do not synchronize",
                "don't synchronize",
                "not at the same time",
                "arrive at different times",
                "different times",
                "stagger",
                "one after another",
                "sequential",
                "wait for",
                "yield",
            ]
            if any(t in desc_lc for t in disable_triggers):
                out["options"]["synchronize_conflicts"] = False

    # speeds
    speeds = obj.get("vehicle_speeds", [])
    if isinstance(speeds, list):
        for s in speeds:
            if not isinstance(s, dict):
                continue
            v = s.get("vehicle")
            sc = s.get("speed_class")
            if v in vset and sc in ("slow", "normal", "fast"):
                out["vehicle_speeds"].append({"vehicle": v, "speed_class": sc})

    # spawn relations
    rels = obj.get("spawn_relations", [])
    if isinstance(rels, list):
        for r in rels:
            if not isinstance(r, dict):
                continue
            typ = r.get("type")
            a = r.get("a")
            b = r.get("b")
            if typ not in ("ahead_of", "behind_of"):
                continue
            if a not in vset or b not in vset or a == b:
                continue
            dist_m = r.get("distance_m", 10.0)
            tol_m = r.get("tolerance_m", 6.0)
            allow_ol = r.get("allow_other_lane", True)
            try:
                dist_m = float(dist_m)
                tol_m = float(tol_m)
            except Exception:
                dist_m, tol_m = 10.0, 6.0
            out["spawn_relations"].append(
                {
                    "type": typ,
                    "a": a,
                    "b": b,
                    "distance_m": max(0.0, dist_m),
                    "tolerance_m": max(0.0, tol_m),
                    "allow_other_lane": bool(allow_ol),
                }
            )

    # lane changes
    lcs = obj.get("lane_changes", [])
    if isinstance(lcs, list):
        for lc in lcs:
            if not isinstance(lc, dict):
                continue
            v = lc.get("vehicle")
            typ = lc.get("type")
            tgt = lc.get("target")
            style = lc.get("style", "polite")
            timing = lc.get("timing", "near_conflict")
            if typ != "merge_into_lane_of":
                continue
            if v not in vset or tgt not in vset or v == tgt:
                continue
            if style not in ("cut_off", "polite"):
                style = "polite"
            if timing not in ("near_conflict", "asap"):
                timing = "near_conflict"
            out["lane_changes"].append({"vehicle": v, "type": typ, "target": tgt, "style": style, "timing": timing})

    # Defensive: only keep lane-change macros if the description actually mentions a lane change / merge.
    lane_triggers = [
        "lane change",
        "change lane",
        "changes lane",
        "changing lane",
        "merge into",
        "merge onto",
        "merging",
        "cuts off",
        "cut off",
        "cutoff",
        "swerves into",
        "swerving into",
    ]
    if not any(k in desc_lc for k in lane_triggers):
        out["lane_changes"] = []

    return out


__all__ = [
    "_chat_template",
    "_extract_first_json_object",
    "_llm_generate_json",
    "extract_refinement_constraints",
]
