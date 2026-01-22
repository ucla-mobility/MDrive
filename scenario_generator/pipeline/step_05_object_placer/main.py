import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("This script requires numpy") from e

from .assets import get_asset_bbox, keyword_filter_assets, load_assets
from .csp import (
    _compute_merge_min_s_by_vehicle,
    build_stage2_constraints_prompt,
    CandidatePlacement,
    expand_group_to_actors,
    solve_weighted_csp_with_extension,
    validate_actor_specs,
)
from .constants import LATERAL_TO_M
from .filters import _contains_exact_quote, _should_drop_stage1_entity
from .guardrails import (
    apply_after_turn_segment_corrections,
    apply_in_intersection_segment_corrections,
    build_repair_prompt,
    validate_stage2_output,
)
from .model import generate_with_model
from .nodes import _override_seg_points_with_picked, build_segments_from_nodes, load_nodes
from .parsing import parse_llm_json
from .prompts import build_stage1_prompt, build_stage2_prompt, build_vehicle_segment_summaries
from .spawn import build_motion_waypoints, compute_spawn_from_anchor, resolve_nodes_path
from .viz import visualize
import math
import numpy as np


def run_object_placer(args, model=None, tokenizer=None):
    """
    Main pipeline body, optionally reusing a provided model/tokenizer.
    """
    t_obj_start = time.time()
    stats = getattr(args, "stats", None)

    def _bump(key: str, amount: int = 1) -> None:
        if stats is None:
            return
        stats[key] = int(stats.get(key, 0)) + amount

    # Set default values for optional args that may not be provided by SimpleNamespace
    if not hasattr(args, 'placement_mode'):
        args.placement_mode = "csp"  # default to CSP-based placement
    if not hasattr(args, 'do_sample'):
        args.do_sample = False
    if not hasattr(args, 'temperature'):
        args.temperature = 0.2
    if not hasattr(args, 'top_p'):
        args.top_p = 0.95

    t0 = time.time()
    with open(args.picked_paths, "r", encoding="utf-8") as f:
        picked_payload = json.load(f)

    picked = picked_payload.get("picked", [])
    if not isinstance(picked, list) or not picked:
        raise SystemExit("[ERROR] picked_paths_detailed.json has no 'picked' list.")

    crop_region = picked_payload.get("crop_region")
    nodes_field = picked_payload.get("nodes")
    if not nodes_field:
        raise SystemExit("[ERROR] picked_paths_detailed.json missing 'nodes' field")

    resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
    if not os.path.exists(resolved_nodes_path):
        raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                         f"Tip: pass --nodes-root to resolve relative paths.")
    nodes: Optional[Dict[str, Any]] = None
    all_segments: List[Dict[str, Any]] = []
    seg_by_id: Dict[int, np.ndarray] = {}
    if args.placement_mode == "csp" or args.viz:
        nodes = load_nodes(resolved_nodes_path)
        all_segments = build_segments_from_nodes(nodes)
        seg_by_id = {int(s["seg_id"]): s["points"] for s in all_segments}
        seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)

    all_assets = load_assets(args.carla_assets)

    # Build vehicle segment summaries for LLM
    vehicle_segments = build_vehicle_segment_summaries(picked)
    print(f"[TIMING] object_placer setup (load paths, assets, summaries): {time.time() - t0:.2f}s", flush=True)

    # Load HF model if not provided
    if tokenizer is None or model is None:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        print(f"[TIMING] object_placer model load: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Stage 1: extract entities (or use schema-provided actors)
    # --------------------------
    t_stage1_start = time.time()
    schema_entities = getattr(args, "schema_entities", None)
    if schema_entities is not None:
        print("[INFO] Using schema-provided actors; skipping Stage1 LLM.")
        stage1_obj = {"entities": schema_entities}
    else:
        t_stage1_start = time.time()
        stage1_prompt = build_stage1_prompt(args.description)
        t0 = time.time()
        stage1_text = generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompt=stage1_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"[TIMING] Stage1 LLM generation: {time.time() - t0:.2f}s", flush=True)
        t0 = time.time()
        # Stage 1 parse (with repair if the model didn't output JSON)
        try:
            stage1_obj = parse_llm_json(stage1_text, required_top_keys=["entities"])
        except Exception:
            _bump("object_stage1_json_repair")
            repair_prompt = (
                "Return JSON ONLY with top-level key 'entities' (a list). No prose.\n"
                "If you previously wrote anything else, convert it into the required JSON now.\n\n"
                "RAW OUTPUT:\n" + stage1_text
            )
            repair_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=repair_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            stage1_obj = parse_llm_json(repair_text, required_top_keys=["entities"])
        print(f"[TIMING] Stage1 parse+repair: {time.time() - t0:.2f}s", flush=True)

    entities = stage1_obj.get("entities", [])
    if not isinstance(entities, list):
        raise SystemExit("[ERROR] Stage1: 'entities' must be a list.")

    # Ensure each entity has a unique entity_id (normalize if LLM didn't provide one)
    valid_entity_ids = set()
    for idx, e in enumerate(entities):
        if not e.get("entity_id"):
            e["entity_id"] = f"entity_{idx + 1}"
        valid_entity_ids.add(e["entity_id"])

    
    # Post-filter Stage 1 entities to reduce hallucinations (Fix A + Fix D)
    t0 = time.time()
    dropped_stage1: List[Tuple[str, str, str]] = []
    filtered_entities: List[Dict[str, Any]] = []

    def _repair_evidence_with_llm(ent: Dict[str, Any]) -> None:
        """Best-effort: if Stage1 paraphrased, try to recover an EXACT supporting quote."""
        ev = str(ent.get("evidence") or "").strip()
        mention = str(ent.get("mention") or "").strip()
        if ev and _contains_exact_quote(args.description, ev):
            return
        if mention and _contains_exact_quote(args.description, mention):
            ent["evidence"] = mention
            return

        # One-shot repair: ask the model to point to an exact substring.
        try:
            _bump("object_stage1_evidence_repair")
            prompt = (
                "Return JSON ONLY: {\"evidence\": \"...\"}.\n"
                "The evidence MUST be an EXACT substring (<=20 words) copied from DESCRIPTION that explicitly mentions the actor (not just a location).\n"
                "If you cannot find any supporting substring, return {\"evidence\": \"\"}.\n\n"
                f"ACTOR_KIND: {ent.get('actor_kind','')}\n"
                f"MENTION (may be paraphrase): {mention}\n\n"
                f"DESCRIPTION:\n{args.description}\n"
            )
            txt = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            obj = parse_llm_json(txt, required_top_keys=["evidence"])
            rep = str(obj.get("evidence") or "").strip()
            if rep and _contains_exact_quote(args.description, rep):
                ent["evidence"] = rep
        except Exception:
            return

    def _sanitize_trigger_action(ent: Dict[str, Any]) -> None:
        trigger = ent.get("trigger")
        action = ent.get("action")

        # Trigger sanitization
        if not isinstance(trigger, dict):
            ent["trigger"] = None
        else:
            ttype = str(trigger.get("type", "")).strip()
            if ttype != "distance_to_vehicle":
                ent["trigger"] = None
            else:
                vehicle = str(trigger.get("vehicle", "")).strip()
                m = re.match(r"^(?:Vehicle|vehicle)\s+(\d+)$", vehicle)
                if not m:
                    ent["trigger"] = None
                else:
                    vehicle = f"Vehicle {m.group(1)}"
                    try:
                        distance_m = float(trigger.get("distance_m", 8.0))
                    except Exception:
                        distance_m = 8.0
                    # Clamp to a sane range
                    distance_m = max(1.0, min(50.0, distance_m))
                    evidence = str(trigger.get("evidence", "")).strip()
                    if evidence and not _contains_exact_quote(args.description, evidence):
                        evidence = ""
                    ent["trigger"] = {
                        "type": "distance_to_vehicle",
                        "vehicle": vehicle,
                        "distance_m": distance_m,
                        "evidence": evidence,
                    }

        # Action sanitization
        if not isinstance(action, dict):
            ent["action"] = None
        else:
            atype = str(action.get("type", "")).strip()
            if atype not in ("start_motion", "hard_brake", "lane_change"):
                ent["action"] = None
            else:
                direction = action.get("direction")
                direction = str(direction).strip().lower() if direction is not None else ""
                if direction not in ("left", "right"):
                    direction = None
                target_vehicle = action.get("target_vehicle")
                target_vehicle = str(target_vehicle).strip() if target_vehicle else None
                if target_vehicle and not re.match(r"^Vehicle\s+\d+$", target_vehicle):
                    target_vehicle = None
                ent["action"] = {
                    "type": atype,
                    "direction": direction,
                    "target_vehicle": target_vehicle,
                }

    def _extract_trigger_candidates(description: str) -> List[Dict[str, Any]]:
        patterns = [
            re.compile(
                r"(?i)\b(?:when|once|if)\s+(Vehicle\s+\d+)\s+(?:gets|is)\s+within\s+(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metres)\b"
            ),
            re.compile(
                r"(?i)\b(?:when|once|if)\s+(Vehicle\s+\d+)\s+gets\s+close\b"
            ),
            re.compile(
                r"(?i)\b(?:when|once|if)\s+(Vehicle\s+\d+)\s+approaches\b"
            ),
        ]
        candidates: List[Dict[str, Any]] = []
        for pat in patterns:
            for m in pat.finditer(description):
                vehicle = m.group(1)
                distance_m = 8.0
                if m.lastindex and m.lastindex >= 2:
                    try:
                        distance_m = float(m.group(2))
                    except Exception:
                        distance_m = 8.0
                candidates.append({
                    "vehicle": vehicle,
                    "distance_m": distance_m,
                    "evidence": m.group(0).strip(),
                    "span": m.span(),
                })
        return candidates

    def _find_substring_span(text: str, sub: str) -> Optional[Tuple[int, int]]:
        if not sub:
            return None
        idx = text.find(sub)
        if idx == -1:
            return None
        return idx, idx + len(sub)

    def _sentence_bounds(text: str, idx: int) -> Tuple[int, int]:
        seps = ".!?"
        start = -1
        for sep in seps:
            start = max(start, text.rfind(sep, 0, idx))
        start = 0 if start == -1 else start + 1
        end_positions = [text.find(sep, idx) for sep in seps if text.find(sep, idx) != -1]
        end = min(end_positions) if end_positions else len(text)
        return start, end

    def _apply_trigger_fallback(entities: List[Dict[str, Any]], description: str) -> None:
        candidates = _extract_trigger_candidates(description)
        if not candidates:
            return
        used = set()

        for ent in entities:
            if ent.get("trigger") is not None:
                continue
            if ent.get("action") is None:
                continue
            mention = str(ent.get("mention") or "")
            evidence = str(ent.get("evidence") or "")
            anchor_span = _find_substring_span(description, mention) or _find_substring_span(description, evidence)
            anchor_idx = anchor_span[0] if anchor_span else None

            available = [(i, c) for i, c in enumerate(candidates) if i not in used]
            if not available:
                continue

            picked = None
            picked_idx = None
            if anchor_idx is not None:
                sent_start, sent_end = _sentence_bounds(description, anchor_idx)
                in_sentence = []
                for idx, cand in available:
                    span = cand.get("span", (0, 0))
                    if span[0] >= sent_start and span[1] <= sent_end:
                        in_sentence.append((idx, cand))
                if in_sentence:
                    def _dist(c):
                        span = c.get("span", (0, 0))
                        mid = (span[0] + span[1]) / 2.0
                        return abs(mid - anchor_idx)
                    picked_idx, picked = min(in_sentence, key=lambda ic: _dist(ic[1]))
                    used.add(picked_idx)
                elif len(available) == 1:
                    picked_idx, picked = available[0]
                    used.add(picked_idx)
                else:
                    def _dist(c):
                        span = c.get("span", (0, 0))
                        mid = (span[0] + span[1]) / 2.0
                        return abs(mid - anchor_idx)
                    picked_idx, picked = min(available, key=lambda ic: _dist(ic[1]))
                    used.add(picked_idx)
            else:
                if len(available) == 1 and len(entities) == 1:
                    picked_idx, picked = available[0]
                    used.add(picked_idx)

            if picked:
                ent["trigger"] = {
                    "type": "distance_to_vehicle",
                    "vehicle": picked["vehicle"],
                    "distance_m": float(picked["distance_m"]),
                    "evidence": picked["evidence"],
                }
                ent_id = ent.get("entity_id", "entity")
                print(f"[INFO] Stage1 trigger fallback: {ent_id} -> {picked['vehicle']} @ {picked['distance_m']}m")

    def _is_moving_entity(ent: Dict[str, Any]) -> bool:
        kind = str(ent.get("actor_kind") or "").strip()
        motion_hint = str(ent.get("motion_hint") or "").strip().lower()
        speed_hint = str(ent.get("speed_hint") or "").strip().lower()
        action = ent.get("action")
        if isinstance(action, dict) and action.get("type") in ("start_motion", "hard_brake", "lane_change"):
            return True
        if motion_hint in ("crossing", "follow_lane"):
            return True
        if speed_hint in ("slow", "normal", "fast", "erratic"):
            return True
        if motion_hint == "static" or speed_hint == "stopped":
            return False
        return kind in ("walker", "cyclist", "npc_vehicle")

    def _default_trigger_vehicle(ent: Dict[str, Any]) -> str:
        vehicle = str(ent.get("affects_vehicle") or "").strip()
        if vehicle.lower() in ("ego", "ego vehicle", "ego_vehicle"):
            return "Vehicle 1"
        match = re.search(r"(\d+)", vehicle)
        if match:
            return f"Vehicle {match.group(1)}"
        return "Vehicle 1"

    def _apply_default_movement_triggers(entities: List[Dict[str, Any]]) -> None:
        for ent in entities:
            if not _is_moving_entity(ent):
                continue
            if ent.get("trigger") is None:
                ent["trigger"] = {
                    "type": "distance_to_vehicle",
                    "vehicle": _default_trigger_vehicle(ent),
                    "distance_m": 8.0,
                    "evidence": "",
                }
            if ent.get("action") is None:
                ent["action"] = {
                    "type": "start_motion",
                    "direction": None,
                    "target_vehicle": None,
                }
            _sanitize_trigger_action(ent)

    for e in entities:
        _repair_evidence_with_llm(e)
        _sanitize_trigger_action(e)

        drop, reason = _should_drop_stage1_entity(e, args.description)
        if drop:
            dropped_stage1.append((str(e.get("entity_id")), reason, str(e.get("mention") or "")))
            continue
        filtered_entities.append(e)
    _apply_trigger_fallback(filtered_entities, args.description)
    if dropped_stage1:
        for ent_id, reason, mention in dropped_stage1[:50]:
            print(f"[WARNING] Stage1 drop: {ent_id}: {reason}; mention='{mention[:60]}'")
    entities = filtered_entities
    _apply_default_movement_triggers(entities)
    valid_entity_ids = set(str(e.get("entity_id")) for e in entities if e.get("entity_id"))
    print(f"[TIMING] Stage1 entity filtering+evidence repair: {time.time() - t0:.2f}s", flush=True)
    print(f"[TIMING] Stage1 total: {time.time() - t_stage1_start:.2f}s", flush=True)

    print(f"[INFO] Stage1 extracted {len(entities)} entities: {list(valid_entity_ids)}")
    # Debug: show Stage 1 entity details including motion_hint and when (route phase)
    for e in entities:
        print(f"  - {e.get('entity_id')}: kind={e.get('actor_kind')}, motion_hint={e.get('motion_hint')}, when={e.get('when')}, mention='{e.get('mention', '')[:50]}'")

    # Keyword synonyms for better asset matching
    t0 = time.time()
    KEYWORD_SYNONYMS = {
        "cyclist": ["bike", "bicycle", "crossbike"],
        "bicyclist": ["bike", "bicycle", "crossbike"],
        "biker": ["bike", "bicycle", "crossbike", "motorcycle"],
        "motorcyclist": ["motorcycle", "harley", "yamaha", "kawasaki"],
        "pedestrian": ["pedestrian", "walker", "person"],
        "person": ["pedestrian", "walker"],
        "cone": ["cone", "trafficcone", "constructioncone"],
        "cones": ["cone", "trafficcone", "constructioncone"],
        "traffic cone": ["cone", "trafficcone", "constructioncone"],
        "barrier": ["barrier", "streetbarrier", "construction"],
        "truck": ["truck", "firetruck", "cybertruck", "pickup"],
        "police": ["police", "charger", "crown"],
        "ambulance": ["ambulance"],
        "firetruck": ["firetruck"],
    }
    OCCLUSION_HINTS = (
        "obstruct", "obstructs", "obstructing", "occlude", "occluding", "visibility",
        "view", "block", "blocking", "obscure", "obscuring", "blind", "line of sight",
    )
    LARGE_STATIC_TOKENS = (
        "streetbarrier", "barrier", "chainbarrier", "busstop", "advertisement",
        "container", "clothcontainer", "glasscontainer",
    )
    SMALL_STATIC_TOKENS = (
        "barrel", "cone", "trafficcone", "bin", "box", "trash", "garbage", "bag", "bottle", "can",
    )
    SPECIFIC_STATIC_TOKENS = LARGE_STATIC_TOKENS + SMALL_STATIC_TOKENS

    def _has_occlusion_hint(mention: str, evidence: str, description: str, static_count: int) -> bool:
        text = f"{mention} {evidence}".lower()
        if any(h in text for h in OCCLUSION_HINTS):
            return True
        if static_count == 1:
            dlow = str(description or "").lower()
            return any(h in dlow for h in OCCLUSION_HINTS)
        return False

    def _mentions_specific_static_prop(text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low for t in SPECIFIC_STATIC_TOKENS)

    def _asset_matches_tokens(asset, tokens) -> bool:
        hay = " ".join([asset.asset_id.lower()] + asset.tags)
        return any(t in hay for t in tokens)

    def _asset_area(asset) -> float:
        if not asset.bbox:
            return 0.0
        return float(asset.bbox.length * asset.bbox.width)

    def _merge_assets(primary, secondary, limit: int = 12):
        seen = set()
        out = []
        for a in list(primary) + list(secondary):
            if a.asset_id in seen:
                continue
            seen.add(a.asset_id)
            out.append(a)
            if len(out) >= limit:
                break
        return out

    large_vehicle_assets = [a for a in all_assets if a.category == "vehicle" and a.bbox]
    large_vehicle_assets.sort(key=_asset_area, reverse=True)
    top_vehicle_occluders = large_vehicle_assets[:6]

    static_like_count = sum(
        1 for e in entities if str(e.get("actor_kind", "")) in ("static_prop", "parked_vehicle")
    )

    # For each entity, build small asset option list (keyed by entity_id)
    per_entity_options: Dict[str, List[Dict[str, Any]]] = {}
    for idx, e in enumerate(entities):
        entity_id = e.get("entity_id", f"entity_{idx+1}")
        mention = str(e.get("mention", f"entity_{idx+1}"))
        evidence = str(e.get("evidence", ""))
        kind = str(e.get("actor_kind", "static_prop"))
        low = mention.lower()
        occlusion_hint = False
        if kind in ("static_prop", "parked_vehicle"):
            occlusion_hint = _has_occlusion_hint(mention, evidence, args.description, static_like_count)

        # Extract keywords: split mention into words and check synonyms
        kws = set()
        words = [w.strip(".,!?") for w in low.split()]
        
        # Add all meaningful words from mention
        for word in words:
            if len(word) > 2:  # Skip tiny words
                kws.add(word)
            # Expand synonyms
            if word in KEYWORD_SYNONYMS:
                kws.update(KEYWORD_SYNONYMS[word])
        
        # Also check multi-word phrases
        for phrase, synonyms in KEYWORD_SYNONYMS.items():
            if phrase in low:
                kws.update(synonyms)

        # Add kind-based keywords (always, not just as fallback)
        if kind == "walker":
            categories = ["walker"]
            kws.update(["pedestrian", "walker"])
        elif kind == "cyclist":
            categories = ["vehicle"]
            kws.update(["bike", "bicycle", "crossbike"])
        elif kind in ("parked_vehicle", "npc_vehicle"):
            categories = ["vehicle"]
            # Keep mention-based keywords; add generic fallbacks only if empty
            if not any(w in kws for w in ["car", "truck", "bus", "van", "vehicle"]):
                kws.update(["car", "vehicle"])
        else:
            categories = ["static"]
            kws.update(["prop", "static"])
            if occlusion_hint:
                kws.update(["barrier", "streetbarrier", "chainbarrier", "busstop", "advertisement", "container"])

        kws_list = list(kws)
        options = keyword_filter_assets(all_assets, kws_list, categories=categories, k=12)

        # last-resort fallback options
        if not options:
            # choose a small default set by category
            options = keyword_filter_assets(all_assets, ["vehicle"], categories=categories, k=12) or all_assets[:12]

        if occlusion_hint and kind == "static_prop":
            text = f"{mention} {evidence}"
            if not _mentions_specific_static_prop(text):
                options = _merge_assets(top_vehicle_occluders, options)
            large = [a for a in options if _asset_matches_tokens(a, LARGE_STATIC_TOKENS)]
            if large:
                options = _merge_assets(large, options)
            else:
                options.sort(key=lambda a: (_asset_matches_tokens(a, SMALL_STATIC_TOKENS), a.asset_id))
        if occlusion_hint and kind == "parked_vehicle":
            options = _merge_assets(top_vehicle_occluders, options)
            options.sort(key=_asset_area, reverse=True)

        # Key by entity_id for Stage 2
        per_entity_options[entity_id] = [
            {"asset_id": a.asset_id, "category": a.category, "tags": a.tags[:6]} for a in options
        ]
    print(f"[TIMING] asset matching for {len(entities)} entities: {time.time() - t0:.2f}s", flush=True)

    ego_spawns: List[Dict[str, Any]] = []

    # Handle empty entities case early
    if not entities:
        print("[INFO] No entities extracted in Stage 1. Skipping Stage 2.")
        actors = []
        stage2_obj = {"actors": []}
    else:
        # --------------------------
        # Stage 2: resolve anchors
        # --------------------------
        t_stage2_start = time.time()

        if args.placement_mode == "llm_anchor":
            stage2_prompt = build_stage2_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print("\n[DEBUG] Stage2 raw output (full):\n" + stage2_text + "\n", flush=True)

            try:
                stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actors"])
            except ValueError as exc:
                _bump("object_stage2_json_repair")
                repair_prompt = build_repair_prompt(stage2_text, [f"JSON parse failed: {exc}"])
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actors"])
            actors = stage2_obj.get("actors", [])
            if not isinstance(actors, list):
                raise SystemExit("[ERROR] Stage2: 'actors' must be a list.")

            errs = validate_stage2_output(actors, vehicle_segments)
            if errs:
                _bump("object_stage2_validation_repair")
                repair_prompt = build_repair_prompt(stage2_text, errs)
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actors"])
                actors = stage2_obj.get("actors", [])
                if not isinstance(actors, list):
                    raise SystemExit("[ERROR] Stage2 repair: 'actors' must be a list.")
            print(f"[TIMING] Stage2 (llm_anchor mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)
        else:
            # CSP mode: LLM emits symbolic preferences; solver chooses anchors.
            # We need geometry (seg_by_id) to enumerate candidates; build it here.
            t0 = time.time()
            if nodes is None:
                nodes = load_nodes(resolved_nodes_path)
                all_segments = build_segments_from_nodes(nodes)
                seg_by_id = {int(s["seg_id"]): s["points"] for s in all_segments}
                seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
            merge_min_s_by_vehicle = _compute_merge_min_s_by_vehicle(picked_payload, picked, seg_by_id)
            print(f"[TIMING] Stage2 load nodes+build segments: {time.time() - t0:.2f}s", flush=True)

            t0 = time.time()
            stage2_prompt = build_stage2_constraints_prompt(args.description, vehicle_segments, entities, per_entity_options)
            stage2_text = generate_with_model(
                model=model,
                tokenizer=tokenizer,
                prompt=stage2_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"[TIMING] Stage2 LLM generation: {time.time() - t0:.2f}s", flush=True)
            print("\n[DEBUG] Stage2(CSP) raw output (full):\n" + stage2_text + "\n", flush=True)

            # Parse with a simple repair-on-failure
            t0 = time.time()
            try:
                stage2_obj = parse_llm_json(stage2_text, required_top_keys=["actor_specs"])
            except Exception:
                _bump("object_stage2_json_repair")
                repair_prompt = (
                    "Return JSON ONLY with top-level key 'actor_specs' (a list). No prose.\n"
                    "If needed, convert your previous output into the required JSON now.\n\n"
                    "RAW OUTPUT:\n" + stage2_text
                )
                repair_text = generate_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=repair_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                stage2_obj = parse_llm_json(repair_text, required_top_keys=["actor_specs"])
            print(f"[TIMING] Stage2 parse+repair: {time.time() - t0:.2f}s", flush=True)

            actor_specs_raw = stage2_obj.get("actor_specs", [])
            actor_specs, warns = validate_actor_specs(actor_specs_raw, entities, per_entity_options, picked=picked)
            for w in warns[:50]:
                print("[WARNING] Stage2(CSP): " + w)

            # Extract ego vehicle spawn positions BEFORE CSP solve so CSP can enforce buffer constraints
            ego_spawns = []
            for p in picked:
                vehicle_name = p.get("vehicle", "")
                sig = p.get("signature", {}) if isinstance(p.get("signature"), dict) else {}
                entry = sig.get("entry", {}) if isinstance(sig.get("entry"), dict) else {}
                
                if entry and isinstance(entry, dict):
                    entry_point = entry.get("point", {})
                    if isinstance(entry_point, dict):
                        ego_x = entry_point.get("x")
                        ego_y = entry_point.get("y")
                        heading = entry.get("heading_deg", 0.0)
                        
                        if ego_x is not None and ego_y is not None:
                            ego_spawns.append({
                                "vehicle": vehicle_name,
                                "spawn": {
                                    "x": float(ego_x),
                                    "y": float(ego_y),
                                    "yaw_deg": float(heading)
                                }
                            })

            if not actor_specs:
                actors = []
            else:
                t0 = time.time()
                print(f"[TIMING] Starting CSP solve with {len(actor_specs)} actor specs...", flush=True)
                # Use iterative CSP with path extension - extends paths on-demand when distance constraints can't be met
                # Returns expanded crop_region to include extended path areas
                chosen, dbg, crop_region = solve_weighted_csp_with_extension(
                    actor_specs, picked, seg_by_id, crop_region,
                    all_segments=all_segments,
                    nodes=nodes,
                    merge_min_s_by_vehicle=merge_min_s_by_vehicle,
                    min_sep_scale=1.0,
                    max_extension_iterations=3,
                    ego_spawns=ego_spawns,
                )
                print(f"[TIMING] CSP solve (with extension): {time.time() - t0:.2f}s", flush=True)
                print("[INFO] CSP solve debug: " + json.dumps(dbg, indent=2))

                actors = []
                for spec in actor_specs:
                    sid = str(spec["id"])
                    cand = chosen.get(sid)
                    if cand is None:
                        continue

                    # Get bounding box info if available
                    asset_id = spec["asset_id"]
                    bbox = get_asset_bbox(asset_id)
                    bbox_info = None
                    if bbox:
                        bbox_info = {
                            "length": bbox.length,
                            "width": bbox.width,
                            "height": bbox.height,
                        }

                    base_actor = {
                        "id": sid,
                        "semantic": spec.get("semantic", sid),
                        "category": spec["category"],
                        "asset_id": asset_id,
                        "trigger": spec.get("trigger"),
                        "action": spec.get("action"),
                        "placement": {
                            "target_vehicle": f"Vehicle {cand.vehicle_num}" if cand.vehicle_num > 0 else None,
                            "segment_index": int(cand.segment_index),
                            "s_along": float(cand.s_along),
                            "lateral_relation": str(cand.lateral_relation),
                            "seg_id": int(cand.seg_id),  # Store seg_id directly for opposite-lane NPCs
                        },
                        "motion": spec.get("motion", {"type": "static", "speed_profile": "normal"}),
                        "confidence": float(spec.get("confidence", 0.6)),
                        "csp": {
                            "base_score": float(cand.base_score),
                            "path_s_m": float(cand.path_s_m),
                            "relations": spec.get("relations", []),
                        },
                        "bbox": bbox_info,  # Include bounding box if available
                    }

                    expanded = expand_group_to_actors(base_actor, spec, cand, seg_by_id)
                    # Validate expansion: check if quantity matches expected
                    expected_qty = int(spec.get("quantity", 1))
                    if len(expanded) < expected_qty:
                        print(f"[WARNING] {sid}: expected {expected_qty} actors but only got {len(expanded)} "
                              f"(group_pattern={spec.get('group_pattern')}, seg_id={cand.seg_id})", flush=True)
                    actors.extend(expanded)
            print(f"[TIMING] Stage2 (CSP mode) total: {time.time() - t_stage2_start:.2f}s", flush=True)


    # --------------------------
    # Geometry reconstruction
    # --------------------------
    t0 = time.time()
    resolved_nodes_path = resolve_nodes_path(args.picked_paths, str(nodes_field), args.nodes_root)
    if not os.path.exists(resolved_nodes_path):
        raise SystemExit(f"[ERROR] nodes path not found: {resolved_nodes_path}\n"
                         f"Tip: pass --nodes-root to resolve relative paths.")

    nodes = load_nodes(resolved_nodes_path)
    all_segments = build_segments_from_nodes(nodes)
    seg_by_id: Dict[int, np.ndarray] = {int(s["seg_id"]): s["points"] for s in all_segments}
    # If paths were refined (start/end trimming or synthetic segments), prefer those polylines.
    seg_by_id = _override_seg_points_with_picked(picked, seg_by_id)
    print(f"[TIMING] geometry reconstruction: {time.time() - t0:.2f}s", flush=True)

    # --------------------------
    # Convert anchors -> world
    # --------------------------
    t0 = time.time()
    # Guardrail: if Stage1 said "after_turn" but Stage2 placed the actor on a turning connector,
    # shift it onto the inferred post-turn (exit) segment before spawning.
    apply_after_turn_segment_corrections(actors, stage1_obj.get("entities", []), picked, seg_by_id)
    # Guardrail: if Stage1 said "in_intersection", keep it on a turn-connector segment.
    apply_in_intersection_segment_corrections(actors, stage1_obj.get("entities", []), picked, seg_by_id)

    # Guardrail: re-anchor actors to Stage1 affects_vehicle when drifted.
    stage1_affects = {
        str(e.get("entity_id")): str(e.get("affects_vehicle", "") or "").strip()
        for e in stage1_obj.get("entities", [])
        if e.get("entity_id")
    }
    filtered_actors: List[Dict[str, Any]] = []
    for actor in actors:
        ent_id = str(actor.get("entity_id") or actor.get("id") or "")
        intended = stage1_affects.get(ent_id, "")
        placement = actor.get("placement", {}) if isinstance(actor.get("placement", {}), dict) else {}
        placed_tv = str(placement.get("target_vehicle") or "").strip()
        if intended and intended.lower() not in ("", "none", "unknown") and placed_tv not in (intended, ""):
            # Try to re-anchor to intended vehicle's first segment midpoint
            picked_entry = next((p for p in picked if p.get("vehicle") == intended), None)
            if picked_entry:
                seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
                segs = (picked_entry.get("signature", {}) or {}).get("segments_detailed", [])
                if seg_ids:
                    placement = dict(placement)
                    placement["target_vehicle"] = intended
                    placement["segment_index"] = 1
                    placement["seg_id"] = seg_ids[0]
                    placement["s_along"] = 0.2
                    placement.setdefault("lateral_relation", "center")
                    actor = dict(actor)
                    actor["placement"] = placement
                    print(
                        f"[WARNING] Re-anchoring actor {ent_id} to {intended} (was {placed_tv or 'none'})",
                        flush=True,
                    )
                else:
                    print(
                        f"[WARNING] Could not re-anchor actor {ent_id}: no segments for {intended}",
                        flush=True,
                    )
                    continue
        filtered_actors.append(actor)
    actors = filtered_actors

    # Expand quantity>1 entities for llm_anchor placement mode.
    if args.placement_mode == "llm_anchor":
        stage1_entities = stage1_obj.get("entities", [])
        entity_by_id = {
            str(e.get("entity_id")): e for e in stage1_entities if e.get("entity_id")
        }

        def _segment_id_for_actor(actor: Dict[str, Any]) -> Optional[int]:
            placement = actor.get("placement", {}) if isinstance(actor.get("placement", {}), dict) else {}
            seg_id = placement.get("seg_id")
            if seg_id is not None:
                return int(seg_id)
            tv = placement.get("target_vehicle")
            try:
                seg_idx = int(placement.get("segment_index", 0))
            except Exception:
                seg_idx = 0
            if not tv or seg_idx <= 0:
                return None
            picked_entry = next((p for p in picked if p.get("vehicle") == tv), None)
            if not picked_entry:
                return None
            seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
            if not isinstance(seg_ids, list) or seg_idx > len(seg_ids):
                return None
            return int(seg_ids[seg_idx - 1])

        def _normalize_group_pattern(ent: Dict[str, Any], qty: int) -> Optional[Dict[str, Any]]:
            if qty <= 1:
                return None
            patt = str(ent.get("group_pattern", "unknown"))
            if patt == "diagonal":
                patt = "along_lane"
            if patt not in ("across_lane", "along_lane", "scatter", "unknown"):
                patt = "along_lane"
            sl = ent.get("start_lateral")
            el = ent.get("end_lateral")
            if sl is not None and str(sl) not in LATERAL_TO_M:
                sl = None
            if el is not None and str(el) not in LATERAL_TO_M:
                el = None
            return {"pattern": patt, "start_lateral": sl, "end_lateral": el, "spacing_bucket": "auto"}

        expanded: List[Dict[str, Any]] = []
        for actor in actors:
            entity_id = str(actor.get("entity_id") or actor.get("id") or "")
            if entity_id:
                actor.setdefault("id", entity_id)
            ent = entity_by_id.get(entity_id)
            if not ent:
                expanded.append(actor)
                continue

            if "trigger" not in actor and ent.get("trigger") is not None:
                actor["trigger"] = ent.get("trigger")
            if "action" not in actor and ent.get("action") is not None:
                actor["action"] = ent.get("action")

            try:
                qty = int(ent.get("quantity", 1))
            except Exception:
                qty = 1
            qty = max(1, qty)
            if qty <= 1:
                expanded.append(actor)
                continue

            seg_id = _segment_id_for_actor(actor)
            if seg_id is None or seg_id not in seg_by_id:
                print(
                    f"[WARNING] {entity_id}: missing seg_id for group expansion; "
                    f"expected {qty} actors, placing 1",
                    flush=True,
                )
                expanded.append(actor)
                continue

            placement = actor.get("placement", {})
            if isinstance(placement, dict):
                placement["seg_id"] = int(seg_id)
                actor["placement"] = placement

            spec = dict(ent)
            spec["id"] = entity_id
            spec["quantity"] = qty
            spec["group_pattern"] = _normalize_group_pattern(ent, qty)
            spec["category"] = actor.get("category", "static")
            spec["actor_kind"] = ent.get("actor_kind", "static_prop")
            spec["asset_id"] = actor.get("asset_id")

            chosen = CandidatePlacement(
                vehicle_num=0,
                segment_index=int(placement.get("segment_index", 1) or 1),
                seg_id=int(seg_id),
                s_along=float(placement.get("s_along", 0.5)),
                lateral_relation=str(placement.get("lateral_relation", "center") or "center"),
                x=0.0,
                y=0.0,
                yaw_deg=0.0,
                path_s_m=0.0,
                base_score=0.0,
            )
            expanded_children = expand_group_to_actors(actor, spec, chosen, seg_by_id)
            if len(expanded_children) < qty:
                print(
                    f"[WARNING] {entity_id}: expected {qty} actors but only got {len(expanded_children)} "
                    f"(group_pattern={spec.get('group_pattern')}, seg_id={seg_id})",
                    flush=True,
                )
            expanded.extend(expanded_children)
        actors = expanded

    actors_world: List[Dict[str, Any]] = []
    for a in actors:
        placement = a["placement"]
        tv = placement.get("target_vehicle")
        seg_idx = int(placement.get("segment_index", 1))  # 1-based
        s_along = float(placement["s_along"])
        lat_rel = placement["lateral_relation"]
        
        # Check if seg_id is directly specified (for opposite-lane NPCs)
        direct_seg_id = placement.get("seg_id")
        
        if direct_seg_id is not None:
            # Direct seg_id: use it directly without looking up picked paths
            seg_id = int(direct_seg_id)
            seg_pts = seg_by_id.get(seg_id)
        else:
            # Standard lookup via target_vehicle and picked paths
            # Find seg_id from the picked path signature order
            # vehicle_segments contains seg_id list in order via picked signature; easiest: pull from picked itself
            picked_entry = next((p for p in picked if p.get("vehicle") == tv), None)
            if not picked_entry:
                continue
            seg_ids = (picked_entry.get("signature", {}) or {}).get("segment_ids", [])
            if not isinstance(seg_ids, list) or seg_idx < 1 or seg_idx > len(seg_ids):
                continue
            seg_id = int(seg_ids[seg_idx - 1])

            seg_pts = seg_by_id.get(seg_id)
            if seg_pts is None:
                # fall back to polyline_sample if present
                segs_det = (picked_entry.get("signature", {}) or {}).get("segments_detailed", [])
                det = next((d for d in segs_det if int(d.get("seg_id", -1)) == seg_id), None)
                if det and isinstance(det.get("polyline_sample"), list) and det["polyline_sample"]:
                    seg_pts = np.array([[p["x"], p["y"]] for p in det["polyline_sample"]], dtype=float)

        if seg_pts is None:
            print(f"[WARNING] Missing segment geometry for seg_id={seg_id}; skipping actor {a.get('id')}")
            continue

        spawn = compute_spawn_from_anchor(seg_pts, s_along, lat_rel, placement.get("lateral_offset_m"))
        motion = a.get("motion", {}) if isinstance(a.get("motion", {}), dict) else {"type": "static"}
        # Let motion builder know anchor s for some types
        motion.setdefault("anchor_s_along", s_along)
        
        # Pass start_lateral for crossing direction inference
        motion.setdefault("start_lateral", lat_rel)

        # category normalization
        cat = str(a.get("category", "")).lower()
        if cat not in ("vehicle", "walker", "static", "cyclist"):
            # derive from asset category if needed
            # (we don't strictly enforce this)
            cat = "static"

        wps = build_motion_waypoints(motion, cat, spawn, seg_pts)
        if isinstance(wps, list) and wps and isinstance(wps[0], dict) and "x" in wps[0] and "y" in wps[0]:
            # Align spawn to the first waypoint so spawn == trajectory start.
            spawn = {
                "x": float(wps[0]["x"]),
                "y": float(wps[0]["y"]),
                "yaw_deg": float(wps[0].get("yaw_deg", spawn.get("yaw_deg", 0.0))),
            }

        actors_world.append({
            **a,
            "resolved": {
                "seg_id": seg_id,
                "nodes_path": resolved_nodes_path,
            },
            "spawn": spawn,
            "world_waypoints": wps,
        })
    print(f"[TIMING] anchor -> world conversion: {time.time() - t0:.2f}s", flush=True)

    # Enforce non-overlapping spawns using asset bounding boxes.
    def _approx_width(a: Dict[str, Any]) -> float:
        bbox = get_asset_bbox(str(a.get("asset_id", "")))
        if bbox and bbox.width and bbox.width > 0:
            return float(bbox.width)
        cat = str(a.get("category", "")).lower()
        # Fallback widths (meters)
        if cat == "vehicle":
            return 1.8
        if cat == "cyclist":
            return 0.6
        if cat == "walker":
            return 0.5
        return 0.4

    def _total_length(seg_pts: np.ndarray) -> float:
        if not isinstance(seg_pts, np.ndarray) or len(seg_pts) < 2:
            return 0.0
        diffs = seg_pts[1:] - seg_pts[:-1]
        return float(np.linalg.norm(diffs, axis=1).sum())

    def _reposition_forward(idx: int, meters: float) -> bool:
        """Shift actor forward along its segment by given meters; recompute spawn/waypoints."""
        try:
            actor = actors_world[idx]
            seg_id = int((actor.get("resolved") or {}).get("seg_id", -1))
            seg_pts = seg_by_id.get(seg_id)
            if seg_pts is None:
                return False
            total = _total_length(seg_pts)
            s_delta = meters / max(total, 1e-6)
            motion = actor.get("motion", {}) if isinstance(actor.get("motion", {}), dict) else {"type": "static"}
            # Increase anchor position modestly
            anchor_s = float(motion.get("anchor_s_along", 0.5))
            new_s = min(0.98, max(0.0, anchor_s + s_delta))
            motion["anchor_s_along"] = new_s
            lateral = str(motion.get("start_lateral", "center") or "center").lower()
            spawn = compute_spawn_from_anchor(seg_pts, new_s, lateral)
            cat = str(actor.get("category", "")).lower()
            wps = build_motion_waypoints(motion, cat, spawn, seg_pts)
            if isinstance(wps, list) and wps and isinstance(wps[0], dict):
                actor["spawn"] = {
                    "x": float(wps[0]["x"]),
                    "y": float(wps[0]["y"]),
                    "yaw_deg": float(wps[0].get("yaw_deg", spawn.get("yaw_deg", 0.0))),
                }
                actor["world_waypoints"] = wps
                actor["motion"] = motion
                return True
        except Exception:
            return False
        return False

    changed = True
    passes = 0
    while changed and passes < 5:
        changed = False
        passes += 1
        for i in range(len(actors_world)):
            ai = actors_world[i]
            xi, yi = float(ai.get("spawn", {}).get("x", 0.0)), float(ai.get("spawn", {}).get("y", 0.0))
            wi = _approx_width(ai)
            ri = wi * 0.5
            for j in range(i + 1, len(actors_world)):
                aj = actors_world[j]
                xj, yj = float(aj.get("spawn", {}).get("x", 0.0)), float(aj.get("spawn", {}).get("y", 0.0))
                wj = _approx_width(aj)
                rj = wj * 0.5
                d = math.hypot(xi - xj, yi - yj)
                threshold = ri + rj + 0.2  # small margin
                if d < threshold:
                    # Push the later actor forward along its segment by the overlap amount
                    push_m = max(0.5, threshold - d)
                    if _reposition_forward(j, push_m):
                        changed = True
                        # Update positions after move
                        xj, yj = float(actors_world[j]["spawn"]["x"]), float(actors_world[j]["spawn"]["y"])
    if passes > 0:
        print(f"[INFO] Non-overlap enforcement passes: {passes}")

    # Validation: Check if total placed actors matches expected from Stage 1
    stage1_entities = stage1_obj.get("entities", [])
    total_expected = sum(int(e.get("quantity", 1)) for e in stage1_entities)
    total_placed = len(actors_world)
    if total_placed < total_expected:
        print(f"[WARNING] VALIDATION FAILED: Stage 1 expected {total_expected} total actors "
              f"but only {total_placed} were placed. Check group expansion logs above.", flush=True)
        for e in stage1_entities:
            qty = int(e.get("quantity", 1))
            if qty > 1:
                eid = e.get("entity_id", "unknown")
                # Count how many actors have this entity as base
                placed_for_entity = sum(1 for a in actors_world if str(a.get("id", "")).startswith(eid))
                if placed_for_entity < qty:
                    print(f"  - {eid}: expected {qty}, placed {placed_for_entity}", flush=True)

    # Verify actors maintain minimum buffer from ego spawns (buffer enforced in CSP during placement)
    MIN_BUFFER_TO_ACTORS_M = 15.0  # Minimum buffer from ego spawn to any actor
    if ego_spawns:
        violations = []
        for actor in actors_world:
            actor_spawn = actor.get("spawn", {})
            if not isinstance(actor_spawn, dict):
                continue
            ax = actor_spawn.get("x")
            ay = actor_spawn.get("y")
            if ax is None or ay is None:
                continue
            
            for ego in ego_spawns:
                ego_spawn = ego.get("spawn", {})
                ex = ego_spawn.get("x")
                ey = ego_spawn.get("y")
                if ex is None or ey is None:
                    continue
                
                dist = math.hypot(float(ax) - float(ex), float(ay) - float(ey))
                if dist < MIN_BUFFER_TO_ACTORS_M:
                    actor_id = actor.get("id", "actor_?")
                    vehicle_id = ego.get("vehicle", "Vehicle?")
                    violations.append({
                        "actor": actor_id,
                        "vehicle": vehicle_id,
                        "distance": dist,
                        "required": MIN_BUFFER_TO_ACTORS_M
                    })
        
        if violations:
            print(f"[ERROR] {len(violations)} actor(s) violate ego spawn buffer constraint:", flush=True)
            for v in violations:
                print(f"  {v['actor']} only {v['distance']:.1f}m from {v['vehicle']} (min {v['required']}m)", flush=True)

    out_payload = {
        "source_picked_paths": args.picked_paths,
        "nodes": resolved_nodes_path,
        "crop_region": crop_region,
        "ego_picked": picked,
        "ego_spawns": ego_spawns,
        "actors": actors_world,
        "macro_plan": stage1_obj.get("entities", []),
    }

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[INFO] Wrote scene objects to: {args.out} (actors={len(actors_world)})")

    if args.viz:
        t_viz = time.time()
        visualize(
            picked=picked,
            seg_by_id=seg_by_id,
            actors_world=actors_world,
            crop_region=crop_region if isinstance(crop_region, dict) else None,
            out_path=args.viz_out,
            description=args.description,
            show=args.viz_show,
        )
        print(f"[TIMING] visualization: {time.time() - t_viz:.2f}s", flush=True)
    print(f"[TIMING] object_placer total (internal): {time.time() - t_obj_start:.2f}s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--picked-paths", required=True, help="picked_paths_detailed.json")
    ap.add_argument("--carla-assets", required=True, help="carla_assets.json")

    ap.add_argument("--description", required=True, help="Natural-language scene description")
    ap.add_argument("--out", default="scene_objects.json", help="Output IR + placements JSON")
    ap.add_argument("--viz-out", default="scene_objects.png", help="Output visualization image")
    ap.add_argument("--viz", action="store_true", help="Enable visualization")
    ap.add_argument("--viz-show", action="store_true", help="Show plot window (if supported)")

    ap.add_argument("--nodes-root", default=None, help="Optional root to resolve relative nodes path")
    ap.add_argument("--placement-mode", default="csp", choices=["csp","llm_anchor"], help="Placement stage: weighted CSP (solver) or legacy LLM anchors")

    # LLM gen controls
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--do-sample", action="store_true", default=True, help="Enable sampling (default: True)")
    ap.add_argument("--no-sample", dest="do_sample", action="store_false", help="Disable sampling")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.95)

    args = ap.parse_args()
    run_object_placer(args)


if __name__ == "__main__":
    main()
