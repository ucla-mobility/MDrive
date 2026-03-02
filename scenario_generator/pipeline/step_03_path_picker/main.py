import argparse
import json
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .constraints import _build_constraints_prompt, _sanitize_constraints_obj
from .csp import _solve_paths_csp
from .fuzzy import _best_fuzzy_match
from .parsing import _extract_description_from_prompt, _safe_parse_json_object, _safe_parse_model_output
from .viz import _build_segments_minimal, _load_nodes, _plot_paths_together


def pick_paths_with_model(
    prompt: str,
    aggregated_json: str,
    out_picked_json: str,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 2048,
    do_sample: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.95,
    allow_fuzzy_match: bool = False,
    viz: bool = False,
    viz_out: str = "picked_paths_viz.png",
    viz_show: bool = False,
    model_id: Optional[str] = None,
    require_straight: bool = False,
    require_on_ramp: bool = False,
    schema_constraints: Optional[Dict[str, Any]] = None,
    required_relations: Optional[List[Dict[str, Any]]] = None,
):
    """
    Run the path picker using a provided model/tokenizer if given.
    Falls back to loading from model_id when not supplied (keeps CLI behavior).
    """
    if tokenizer is None or model is None:
        if not model_id:
            raise ValueError("model_id is required when model/tokenizer are not provided")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()

    with open(aggregated_json, "r", encoding="utf-8") as f:
        agg = json.load(f)

    candidates = agg.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("[ERROR] aggregated-json has no 'candidates' list.")

    def _generate_text(local_prompt: str, local_max_tokens: Optional[int] = None) -> str:
        import time
        effective_max = local_max_tokens if local_max_tokens is not None else max_new_tokens
        if getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": local_prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if torch.cuda.is_available():
                input_ids = input_ids.to(model.device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            gen_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            input_len = int(input_ids.shape[-1])
        else:
            enc = tokenizer(local_prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                enc = {k: v.to(model.device) for k, v in enc.items()}
            gen_kwargs = enc
            input_len = int(enc["input_ids"].shape[-1])

        # Build generation kwargs; omit temperature/top_p when not sampling to avoid warnings
        gen_config = {
            "max_new_tokens": effective_max,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_config["temperature"] = temperature
            gen_config["top_p"] = top_p

        print(f"[DEBUG] path_picker LLM: prompt_tokens={input_len}, max_new={effective_max}", flush=True)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**gen_kwargs, **gen_config)
        elapsed = time.time() - t0
        out_tokens = out.shape[-1] - input_len
        print(f"[DEBUG] path_picker LLM: done in {elapsed:.1f}s, output_tokens={out_tokens}", flush=True)

        gen_tokens = out[0][input_len:]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True)

    # -------------------------
    # Stage A: constraint extraction (LLM) + deterministic CSP solve
    # If schema_constraints is provided, try deterministic CSP first.
    # -------------------------
    import time as time_module
    parsed: Optional[Dict[str, Any]] = None
    description = _extract_description_from_prompt(prompt)

    if schema_constraints:
        try:
            t0_csp_stage = time_module.time()
            csp_items = _solve_paths_csp(
                schema_constraints,
                candidates,
                description=description,
                require_straight=require_straight,
                require_on_ramp=require_on_ramp,
                lane_counts_by_road=agg.get("lane_counts_by_road") if isinstance(agg, dict) else None,
                skip_evidence_filter=True,
                crop_region=agg.get("crop_region") if isinstance(agg, dict) else None,
                required_relations=required_relations,
            )
            parsed = {"vehicles": csp_items}
            print(f"[TIMING] path_picker schema CSP solve: {time_module.time() - t0_csp_stage:.2f}s", flush=True)
        except Exception as e:
            if required_relations:
                raise
            print(f"[WARNING] Schema constraints CSP solve failed; falling back to LLM. Reason: {e}")

    if parsed is None and description:
        t0_csp_stage = time_module.time()
        constraints_prompt = _build_constraints_prompt(description)
        constraints_text = _generate_text(constraints_prompt)
        print(constraints_text)
        print(f"[TIMING] path_picker constraint extraction LLM: {time_module.time() - t0_csp_stage:.2f}s", flush=True)

        t0_parse = time_module.time()
        constraints_obj = _safe_parse_json_object(constraints_text)
        if constraints_obj:
            # Debug: show extracted lane_change_phase
            for v in constraints_obj.get("vehicles", []):
                lcp = v.get("lane_change_phase")
                man = v.get("maneuver")
                if man == "lane_change":
                    print(f"[DEBUG] {v.get('vehicle')}: maneuver={man}, lane_change_phase={lcp}")
            
            # Deterministic guardrail: drop hallucinated constraints/evidence + dedupe
            constraints_obj = _sanitize_constraints_obj(constraints_obj, description)
            print(f"[TIMING] path_picker parse+sanitize: {time_module.time() - t0_parse:.2f}s", flush=True)

            try:
                t0_csp_solve = time_module.time()
                csp_items = _solve_paths_csp(
                    constraints_obj,
                    candidates,
                    description=description,
                    require_straight=require_straight,
                    require_on_ramp=require_on_ramp,
                    lane_counts_by_road=agg.get("lane_counts_by_road") if isinstance(agg, dict) else None,
                    crop_region=agg.get("crop_region") if isinstance(agg, dict) else None,
                    required_relations=required_relations,
                )
                print(f"[TIMING] path_picker CSP solve: {time_module.time() - t0_csp_solve:.2f}s", flush=True)
                parsed = {"vehicles": csp_items}
            except Exception as e:
                print(f"[WARNING] CSP solve failed; falling back to direct path picking. Reason: {e}")

    # -------------------------
    # Stage B (fallback): direct path picking (legacy behavior)
    # -------------------------
    if parsed is None:
        text = _generate_text(prompt)
        print(text)

        parsed = _safe_parse_model_output(text)
        if not parsed or "vehicles" not in parsed or not isinstance(parsed["vehicles"], list):
            raise SystemExit("[ERROR] Could not parse model output as JSON with top-level 'vehicles' list.")

    with open(aggregated_json, "r", encoding="utf-8") as f:
        agg = json.load(f)

    candidates = agg.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit("[ERROR] aggregated-json has no 'candidates' list.")

    name_to_cand = {c.get("name"): c for c in candidates if isinstance(c, dict) and c.get("name")}
    candidate_names = list(name_to_cand.keys())

    picked: List[Dict[str, Any]] = []
    for item in parsed["vehicles"]:
        if not isinstance(item, dict):
            continue
        veh = item.get("vehicle", "Vehicle")
        req_name = item.get("path_name", "")

        if not req_name:
            m = re.search(r"path_\d{3}[^\s\"']*", json.dumps(item))
            req_name = m.group(0) if m else ""

        if not req_name:
            print(f"[WARNING] Missing path_name for {veh}; skipping.")
            continue

        cand = name_to_cand.get(req_name)
        if cand is None and allow_fuzzy_match:
            alt = _best_fuzzy_match(req_name, candidate_names)
            cand = name_to_cand.get(alt) if alt else None

        if cand is None:
            print(f"[WARNING] Requested path not found: '{req_name}' for {veh}; skipping.")
            continue

        out_entry = {
            "vehicle": veh,
            "name": cand.get("name"),
            "signature": cand.get("signature", {}),
        }
        if "confidence" in item:
            out_entry["confidence"] = item["confidence"]
        picked.append(out_entry)

    out_payload = {
        "source_candidates": aggregated_json,
        "nodes": agg.get("nodes"),
        "crop_region": agg.get("crop_region"),
        "parameters": agg.get("parameters"),
        "picked": picked,
    }

    with open(out_picked_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"[INFO] Wrote {len(picked)} picked paths to: {out_picked_json}")

    if viz:
        nodes_path = agg.get("nodes")
        if not nodes_path:
            raise SystemExit("[ERROR] aggregated-json missing 'nodes' path; cannot visualize.")
        nodes = _load_nodes(nodes_path)
        all_segments = _build_segments_minimal(nodes)
        _plot_paths_together(
            all_segments=all_segments,
            picked=picked,
            crop=agg.get("crop_region"),
            out_path=viz_out,
            show=viz_show,
        )

    return out_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to local model dir (or HF id)")
    ap.add_argument("--prompt-file", required=True, help="Text file containing your prompt")

    ap.add_argument("--max-new-tokens", type=int, default=256)

    ap.add_argument("--do-sample", action="store_true", default=True, help="Enable sampling (default: True)")
    ap.add_argument("--no-sample", dest="do_sample", action="store_false", help="Disable sampling")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.95)

    ap.add_argument("--aggregated-json", required=True, help="legal_paths_detailed.json")
    ap.add_argument("--out-picked-json", required=True, help="Output subset JSON")

    ap.add_argument("--allow-fuzzy-match", action="store_true",
                    help="Allow conservative fuzzy matching when exact name not found.")

    ap.add_argument("--viz", action="store_true", help="If set, generate a combined plot of picked paths.")
    ap.add_argument("--viz-out", type=str, default="picked_paths_viz.png", help="Output image file for viz.")
    ap.add_argument("--viz-show", action="store_true", help="If set, also show the plot window.")

    args = ap.parse_args()

    prompt = open(args.prompt_file, "r", encoding="utf-8").read().strip()

    pick_paths_with_model(
        prompt=prompt,
        aggregated_json=args.aggregated_json,
        out_picked_json=args.out_picked_json,
        model=None,
        tokenizer=None,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_fuzzy_match=args.allow_fuzzy_match,
        viz=args.viz,
        viz_out=args.viz_out,
        viz_show=args.viz_show,
    )


if __name__ == "__main__":
    main()
