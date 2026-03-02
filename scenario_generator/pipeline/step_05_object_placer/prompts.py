import json
from typing import Any, Dict, List

from .constants import LATERAL_RELATIONS


def build_vehicle_segment_summaries(picked: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a small, LLM-friendly summary of each ego vehicle's segments,
    preserving segment order (segment_index is 1-based).
    """
    out: Dict[str, Any] = {}
    for p in picked:
        veh = str(p.get("vehicle", "Vehicle"))
        sig = p.get("signature", {}) if isinstance(p.get("signature", {}), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        mans = sig.get("maneuvers_between", []) if isinstance(sig.get("maneuvers_between", []), list) else []
        summary = []
        for i, s in enumerate(segs):
            if not isinstance(s, dict):
                continue
            entry_man = None
            exit_man = None
            if i > 0 and i - 1 < len(mans):
                entry_man = mans[i - 1]
            if i < len(mans):
                exit_man = mans[i]
            summary.append({
                "segment_index": i + 1,
                "seg_id": s.get("seg_id"),
                "road_id": s.get("road_id"),
                "lane_id": s.get("lane_id"),
                "length_m": s.get("length_m"),
                "start_cardinal4": (s.get("start", {}) or {}).get("cardinal4"),
                "end_cardinal4": (s.get("end", {}) or {}).get("cardinal4"),
                "maneuver_into_segment": entry_man,   # None for first segment
                "maneuver_out_of_segment": exit_man,  # None for last segment
            })
        out[veh] = {
            "path_name": p.get("name"),
            "num_segments": sig.get("num_segments"),
            "segments": summary,
        }
    return out


def fewshot_examples() -> str:
    # Removed: few-shot examples were causing LLM to hallucinate actors not in Stage 1.
    # The schema in the prompt payload is sufficient.
    return ""


def build_stage1_prompt(description: str) -> str:
    """
    Stage 1: entity extraction, without forcing exact segment indices.
    Each entity gets a unique entity_id that Stage 2 MUST use.
    """
    return (
        "You will read a short driving scene description.\n"
        "\n"
        "Extract ALL non-ego actors that should be spawned as obstacles or interacting entities.\n"
        "Assign each entity a UNIQUE entity_id starting from 'entity_1', 'entity_2', etc.\n"
        "For each extracted entity, include an 'evidence' field: an EXACT quote (<=20 words) from the description\n"
        "that clearly mentions the actor (not just the location).\n"
        "\n"
        "\n"
        "CRITICAL RULES - READ CAREFULLY:\n"
        "\n"
        "1. 'Vehicle 1', 'Vehicle 2', 'Vehicle 3', etc. are ALWAYS ego vehicles.\n"
        "   They are NEVER extracted. They already exist in the simulation.\n"
        "\n"
        "2. Descriptions of ego vehicle maneuvers (straight, left, right, turns, continues)\n"
        "   are NOT actors to extract. These just describe what the ego vehicles do.\n"
        "\n"
        "3. Only extract ADDITIONAL actors that are:\n"
        "   - Static props (traffic cones, barriers, debris, boxes)\n"
        "   - Parked/stopped vehicles (parked car, delivery truck blocking lane)\n"
        "   - Pedestrians/walkers (person crossing, pedestrian on sidewalk)\n"
        "   - Cyclists/bicycles\n"
        "   - NPC vehicles that interact (vehicle that yields, cuts in, merges)\n"
        "\n"
        "4. If the description ONLY talks about ego vehicles and their routes,\n"
        "   with no obstacles, pedestrians, or interacting vehicles mentioned,\n"
        "   return: {\"entities\": []}\n"
        "\n"
        "EXAMPLES OF WHAT TO EXTRACT:\n"
        "- '5 traffic cones in the right side of its lane' -> extract as static_prop, quantity=5\n"
        "- 'multiple traffic cones' -> extract as static_prop, quantity=5 (default for 'multiple')\n"
        "- 'a bicyclist in the middle of its lane' -> extract as cyclist, quantity=1\n"
        "- 'a parked truck blocking the lane' -> extract as parked_vehicle, quantity=1\n"
        "- 'a pedestrian crosses the road' -> extract as walker, quantity=1\n"
        "\n"
        "EXAMPLES OF WHAT NOT TO EXTRACT:\n"
        "- 'Vehicle 1 continues straight' -> ego vehicle, do NOT extract\n"
        "- 'Another vehicle turns left' -> ego vehicle (Vehicle 2/3), do NOT extract\n"
        "- 'Vehicle 2 is coming from the same direction' -> ego vehicle, do NOT extract\n"
        "\n"
        "MOTION_HINT DEFINITIONS (use carefully):\n"
        "- 'static': does not move (parked vehicles, cones, barriers)\n"
        "- 'crossing': moves ACROSS the road, perpendicular to traffic (pedestrian crossing street)\n"
        "- 'follow_lane': moves ALONG the road in the lane direction (walks/rides in direction of road)\n"
        "- 'unknown': motion not specified\n"
        "\n"
        "CROSSING_DIRECTION (for crossing motion ONLY):\n"
        "- 'left': crosses FROM right TO left (direction of movement is leftward)\n"
        "- 'right': crosses FROM left TO right (direction of movement is rightward)\n"
        "- null: not specified or not applicable\n"
        "\n"
        "IMPORTANT: 'walks in the direction of the road' = follow_lane (ALONG the road)\n"
        "           'crosses the road' = crossing (ACROSS the road)\n"
        "\n"
        "QUANTITY EXTRACTION:\n"
        "- If a number is given (e.g., '5 cones'), use that number.\n"
        "- 'multiple', 'several', 'some' -> use quantity=5 as default.\n"
        "- 'a few' -> use quantity=3.\n"
        "- 'a', 'an', 'one', or no quantifier -> use quantity=1.\n"
        "\n"
        "GROUP_PATTERN (for quantity > 1):\n"
        "- 'across_lane': objects arranged side-by-side across the lane width\n"
        "- 'along_lane': objects arranged in a line along the direction of travel\n"
        "- 'diagonal': objects arranged diagonally (e.g., 'starting right, ending left')\n"
        "- 'unknown': arrangement not specified\n"
        "\n"
        "WHEN / ROUTE PHASE (CRITICAL for placement):\n"
        "- 'on_approach': on the approach road BEFORE the intersection\n"
        "- 'in_intersection': inside the intersection itself\n"
        "- 'after_turn': immediately after a turning maneuver\n"
        "- 'after_exit': on the EXIT road (after the intersection). Use for 'on Vehicle X's exit road', 'after the intersection', 'on the exit road', 'past the intersection'\n"
        "- 'after_merge': after a merge region on the exit road. Use for 'after the merge', 'past the merge region'\n"
        "- 'unknown': location not specified\n"
        "\n"
        "IMPORTANT: 'On Vehicle 1's exit road' = after_exit (NOT on_approach!)\n"
        "           'after the intersection' = after_exit\n"
        "           'on the approach road' = on_approach\n"
        "\n"
        "SPEED_HINT:\n"
        "- 'stopped': not moving (parked, standing still)\n"
        "- 'slow': moving slowly\n"
        "- 'normal': moving at normal speed\n"
        "- 'fast': moving fast\n"
        "- 'erratic': driving erratically, unpredictably, or aggressively\n"
        "- 'unknown': speed/behavior not specified\n"
        "\n"
        "TRIGGER (optional, only when explicitly stated):\n"
        "- Use when the description says an actor starts AFTER a condition, e.g. \"when Vehicle 1 gets close\".\n"
        "- type: 'distance_to_vehicle'\n"
        "- vehicle: the ego vehicle name (e.g., \"Vehicle 1\")\n"
        "- distance_m: numeric distance in meters; if not specified, use 8.0\n"
        "- evidence: EXACT substring that states the trigger condition\n"
        "- If no trigger is stated, set trigger=null\n"
        "\n"
        "ACTION (optional, only when explicitly stated):\n"
        "- 'start_motion': begin moving along the planned path (use for pedestrians/cyclists that start crossing)\n"
        "- 'hard_brake': full stop (use for NPC vehicles that brake suddenly)\n"
        "- 'lane_change': change lane (use for NPC vehicles that change lanes)\n"
        "- For lane_change, include direction 'left' or 'right' if specified.\n"
        "- If the text says \"into Vehicle X's lane\", set target_vehicle=\"Vehicle X\".\n"
        "- If no explicit action is stated, set action=null\n"
        "\n"
        "DIRECTION_RELATIVE_TO (for NPC vehicles ONLY):\n"
        "- 'same': NPC travels in the same direction as the reference vehicle\n"
        "- 'opposite': NPC travels in the opposite direction (oncoming traffic)\n"
        "- null: not applicable or not specified\n"
        "\n"
        "LANE RELATION GUIDANCE:\n"
        "- If the description says \"lane to the right/left of Vehicle X\", use right_lane/left_lane (adjacent lane center).\n"
        "- Use half_right/half_left for within-lane offsets, not adjacent lanes.\n"
        "- For visual static props described at a lane edge (e.g., cones after a merge), use right_edge/left_edge and arrange along_lane; avoid center unless explicitly blocking.\n"
        "\n"
        "Return JSON ONLY with this schema:\n"
        "IMPORTANT: The \'mention\' field MUST be an EXACT substring copied from the DESCRIPTION (no paraphrase).\n"
        "- Example: if DESCRIPTION says \'cyclist\', do NOT output \'bicyclist\'; copy \'cyclist\'.\n"
        "{\n"
        "  \"entities\": [\n"
        "    {\n"
        "      \"entity_id\": \"entity_1\",\n"
        "      \"mention\": \"...\",\n"
        "      \"evidence\": \"EXACT quote (<=20 words) from description that mentions this actor\",\n"
        "      \"actor_kind\": \"static_prop\" | \"parked_vehicle\" | \"walker\" | \"cyclist\" | \"npc_vehicle\",\n"
        "      \"quantity\": 1,\n"
        "      \"group_pattern\": \"across_lane\" | \"along_lane\" | \"diagonal\" | \"unknown\",\n"
        "      \"start_lateral\": \"right_edge\" | \"half_right\" | \"center\" | \"half_left\" | \"left_edge\" | \"right_lane\" | \"left_lane\" | null,\n"
        "      \"end_lateral\": \"right_edge\" | \"half_right\" | \"center\" | \"half_left\" | \"left_edge\" | \"right_lane\" | \"left_lane\" | null,\n"
        "      \"affects_vehicle\": \"Vehicle 1\" | \"Vehicle 2\" | \"Vehicle 3\" | null,\n"
        "      \"when\": \"on_approach\" | \"after_turn\" | \"in_intersection\" | \"after_exit\" | \"after_merge\" | \"unknown\",\n"
        "      \"lateral_relation\": \"center\" | \"half_right\" | \"right_edge\" | \"right_lane\" | \"half_left\" | \"left_edge\" | \"left_lane\" | \"unknown\",\n"
        "      \"motion_hint\": \"static\" | \"crossing\" | \"follow_lane\" | \"unknown\",\n"
        "      \"crossing_direction\": \"left\" | \"right\" | null,\n"
        "      \"speed_hint\": \"stopped\" | \"slow\" | \"normal\" | \"fast\" | \"erratic\" | \"unknown\",\n"
        "      \"trigger\": {\n"
        "        \"type\": \"distance_to_vehicle\",\n"
        "        \"vehicle\": \"Vehicle 1\",\n"
        "        \"distance_m\": 8.0,\n"
        "        \"evidence\": \"EXACT trigger phrase from description\"\n"
        "      } | null,\n"
        "      \"action\": {\n"
        "        \"type\": \"start_motion\" | \"hard_brake\" | \"lane_change\",\n"
        "        \"direction\": \"left\" | \"right\" | null,\n"
        "        \"target_vehicle\": \"Vehicle 1\" | null\n"
        "      } | null,\n"
        "      \"direction_relative_to\": {\"vehicle\": \"Vehicle 1\", \"direction\": \"same\" | \"opposite\"} | null\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        f"DESCRIPTION:\n{description}\n"
    )


def build_stage2_prompt(
    description: str,
    vehicle_segments: Dict[str, Any],
    entities: List[Dict[str, Any]],
    per_entity_asset_options: Dict[str, List[Dict[str, Any]]],
) -> str:
    """
    Stage 2: resolve entities to concrete segment anchors + pick asset ids.
    Entities come from Stage 1 and each has a unique entity_id.
    """
    # Build list of valid entity_ids for the prompt
    valid_entity_ids = [e.get("entity_id", f"entity_{i+1}") for i, e in enumerate(entities)]
    
    payload = {
        "vehicle_segments": vehicle_segments,
        "entities": entities,
        "valid_entity_ids": valid_entity_ids,
        "lateral_relations_allowed": LATERAL_RELATIONS,
        "asset_options_per_entity": per_entity_asset_options,
        "notes": [
            "CRITICAL: Output AT MOST one actor per entity_id. If an entity seems invalid/unspawnable (e.g., map geometry), omit it.",
            "CRITICAL: Each actor's 'entity_id' field MUST match one of the valid_entity_ids.",
            "CRITICAL: Do NOT invent new actors. Only place the entities provided.",
            "If 'entities' is empty, return {\"actors\": []}.",
            "segment_index is 1-based and must be valid for the target_vehicle.",
            "s_along is a fraction in [0,1] along that segment.",
            "If something is 'after_turn' and the target vehicle has a non-straight maneuver (left/right/u-turn), place it on the post-turn (exit) segment. In the common 3-segment representation (approach, turn-connector, exit), this is typically segment_index=3 (exit), not segment_index=2 (turn connector).",
            "If something is 'after the merge region' or 'after merge', place it on the EXIT segment (segment_index=3 for a 3-segment path) since merges happen on or after the intersection exit. Use s_along > 0.5 for objects farther down the exit road.",
            "For static visual props after a merge, keep lateral_relation at right_edge/left_edge (same lane) so they do not overlap the ego path centerline.",
            "If unsure, still choose the best guess and use lower confidence.",
            "asset_id MUST be copied EXACTLY from the provided options for that entity.",
            "Include a confidence value in [0,1] for each actor; do not omit it.",
        ],
        "required_output_schema": {
            "actors": [
                {
                    "entity_id": "entity_1",
                    "semantic": "<from entity mention>",
                    "category": "vehicle|walker|static",
                    "asset_id": "<from asset_options_per_entity>",
                    "placement": {
                        "frame": "segment",
                        "target_vehicle": "Vehicle X",
                        "segment_index": 1,
                        "s_along": 0.5,
                        "lateral_relation": "center"
                    },
                    "motion": {
                        "type": "static|cross_perpendicular|follow_lane|straight_line",
                        "speed_profile": "stopped|slow|normal|fast|erratic"
                    },
                    "confidence": 0.9
                }
            ]
        }
    }

    return (
        "You are placing non-ego actors into a road network described by picked ego paths.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1) You MUST output EXACTLY one actor for each entity in the 'entities' list.\n"
        "2) Each actor MUST have an 'entity_id' that matches one from valid_entity_ids.\n"
        "3) Do NOT create any actors that are not in the entities list.\n"
        "4) If the entities list is empty, return {\"actors\": []}.\n"
        "\n"
        "For each entity, you must:\n"
        "1) Copy its entity_id to the actor.\n"
        "2) Choose a concrete segment anchor (target_vehicle + segment_index + s_along + lateral_relation).\n"
        "3) Choose an asset_id EXACTLY from the provided options for that entity.\n"
        "4) Provide motion:\n"
        "   - static: for parked/stationary objects.\n"
        "   - cross_perpendicular: for actors crossing the ego lane.\n"
        "   - follow_lane: for actors moving along the lane direction.\n"
        "   - straight_line: for actors moving between two anchors.\n"
        "\n"
        "Return JSON ONLY with top-level key 'actors'.\n"
        "\n"
        "INPUT (JSON):\n"
        + json.dumps(payload, indent=2)
        + "\n\nSCENE DESCRIPTION:\n"
        + description
        + "\n"
    )


__all__ = [
    "build_stage1_prompt",
    "build_stage2_prompt",
    "build_vehicle_segment_summaries",
    "fewshot_examples",
]
