import json
from typing import Any, Dict, List, Optional


def save_prompt_file(
    out_path: str,
    crop,
    nodes_path: str,
    params: Dict[str, Any],
    paths_named: List[Dict[str, Any]],
) -> None:
    """
    Writes a prompt-ready TEXT file (copy-paste into your LLM).

    OPTIMIZED FOR PATH PICKING:
    - The prompt includes a COMPACT candidate list that is easy for an LLM to scan.
    - It intentionally omits heavy per-segment fields (e.g., segments_detailed) because those
      often degrade selection accuracy by adding noise and increasing context length.
    - If you need segment-level placement (e.g., "immediately after the turn"), use the
      machine-readable JSON output (--out-json-detailed) after the path is selected.

    The compact candidate schema is:
      {
        "name": "...",
        "entry": {"cardinal4": "N", "heading_deg": 90, "road_id": 9, "lane_id": 1},
        "exit":  {"cardinal4": "E", "heading_deg": 0,  "road_id": 10,"lane_id": 1},
        "entry_to_exit_turn": "right",
        "maneuvers_between": ["right","straight"],
        "num_segments": 3,
        "length_m": 85.1
      }
    """
    # Build a compact view of candidates for better LLM accuracy.
    compact_candidates: List[Dict[str, Any]] = []
    for c in paths_named:
        sig = c.get("signature", {})
        ent = sig.get("entry", {})
        ex = sig.get("exit", {})
        compact_candidates.append({
            "name": c.get("name", ""),
            "entry": {
                "cardinal4": ent.get("cardinal4", None),
                "heading_deg": ent.get("heading_deg", None),
                "road_id": ent.get("road_id", None),
                "lane_id": ent.get("lane_id", None),
            },
            "exit": {
                "cardinal4": ex.get("cardinal4", None),
                "heading_deg": ex.get("heading_deg", None),
                "road_id": ex.get("road_id", None),
                "lane_id": ex.get("lane_id", None),
            },
            "entry_to_exit_turn": sig.get("entry_to_exit_turn", None),
            "maneuvers_between": sig.get("maneuvers_between", None),
            "num_segments": sig.get("num_segments", None),
            "length_m": sig.get("length_m", None),
        })

    payload = {
        "context": {
            "nodes": nodes_path,
            "crop_region": {"xmin": crop.xmin, "xmax": crop.xmax, "ymin": crop.ymin, "ymax": crop.ymax},
            "parameters": params,
            "num_candidates": len(compact_candidates),
            "turn_frame": params.get("turn_frame", "WORLD_FRAME"),
        },
        "candidates": compact_candidates,
        "output_schema": {
            "vehicles": [
                {"vehicle": "Vehicle 1", "path_name": "path_###__...", "confidence": 0.0}
            ]
        },
    }

    instructions = (
        "You are given a list of candidate vehicle paths in a road network.\n"
        "Each candidate has a NAME plus a compact SIGNATURE describing its entry direction,\n"
        "exit direction, and overall maneuver.\n"
        "\n"
        "TASK:\n"
        "Given a user scenario description (vehicles and their intended motions), choose the\n"
        "best matching candidate path NAME for each moving vehicle.\n"
        "Ignore static or parked objects; only assign paths to moving vehicles.\n"
        "\n"
        "HOW TO INTERPRET THE DESCRIPTION (IMPORTANT):\n"
        "- Vehicles are described relative to roads, directions, and other vehicles.\n"
        "- Phrases like 'coming from X', 'approaching from X', or 'on the X road' constrain\n"
        "  the ENTRY direction of the path.\n"
        "- Phrases like 'onto X', 'to X', or 'leaves onto X' constrain the EXIT direction.\n"
        "- If a vehicle is described as coming from the same direction as another vehicle,\n"
        "  their chosen paths should have the same or compatible entry direction.\n"
        "- If a T-junction is mentioned, the 'main road' refers to the straight-through road\n"
        "  and the 'side road' refers to the terminating branch.\n"
        "\n"
        "PERPENDICULAR / CROSS-TRAFFIC INTERPRETATION (CRITICAL):\n"
        "- 'Perpendicular road to the RIGHT of Vehicle X's approach': if Vehicle X is westbound (W),\n"
        "  the road to their right is the southbound road (entry=S). If Vehicle X is eastbound (E),\n"
        "  the road to their right is the northbound road (entry=N). Apply 90-degree clockwise rotation.\n"
        "- 'Perpendicular road to the LEFT of Vehicle X's approach': apply 90-degree counter-clockwise.\n"
        "- 'Turns right onto Vehicle X's exit road': the vehicle's EXIT direction must match Vehicle X's\n"
        "  exit direction. If Vehicle X exits westbound (W), the turning vehicle must also exit W.\n"
        "- 'Turns left onto Vehicle X's exit road': same logic - EXIT directions must match.\n"
        "\n"
        "PATH SELECTION GUIDELINES:\n"
        "- First match the described source and destination semantics (where the vehicle comes from and goes to).\n"
        "- Then match the maneuver (left, right, straight).\n"
        "- Prefer paths whose entry and exit directions best align with the description,\n"
        "  even if multiple candidates share the same maneuver.\n"
        "- A path that reverses 'from' vs 'onto' semantics is a poor match and should only\n"
        "  be chosen if no better option exists.\n"
        "\n"
        "IMPORTANT RULES:\n"
        "1) Return JSON ONLY (no prose, no markdown, no code fences).\n"
        "2) path_name MUST be copied EXACTLY from one of the provided candidate name strings.\n"
        "3) If ambiguous, still pick the best match and include confidence in [0,1].\n"
        "\n"
        "CANDIDATE PATHS (JSON):\n"
    )

    with open(out_path, "w") as f:
        f.write(instructions)
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"[INFO] Prompt-ready file saved to: {out_path}")

    # Also write a compact prompt template for CONSTRAINT EXTRACTION (optional, for debugging / inspection).
    # This is NOT used programmatically by default; the pipeline may build its own constraints prompt.
    try:
        import os as _os
        _constraints_path = _os.path.join(_os.path.dirname(out_path), "constraints_prompt.txt")
        _constraints_instructions = (
            "You will read a short driving scene description.\n"
            "Extract constraints about MOVING vehicles only (Vehicle 1, Vehicle 2, ...).\n"
            "Do NOT choose path names here; only list constraints.\n"
            "\n"
            "Return JSON ONLY with this schema:\n"
            "{\n"
            "  \"vehicles\": [\n"
            "    {\n"
            "      \"vehicle\": \"Vehicle 1\",\n"
            "      \"maneuver\": \"left\" | \"right\" | \"straight\" | \"unknown\",\n"
            "      \"entry_road\": \"main\" | \"side\" | \"unknown\",\n"
            "      \"exit_road\": \"main\" | \"side\" | \"unknown\"\n"
            "    }\n"
            "  ],\n"
            "  \"constraints\": [\n"
            "    { \"type\": \"same_entry_as\" | \"opposite_entry_of\" | \"same_exit_as\" | \"opposite_exit_of\", \"a\": \"Vehicle X\", \"b\": \"Vehicle Y\" }\n"
            "  ]\n"
            "}\n"
            "\n"
            "Guidance:\n"
            "- If the description says \"coming from the same direction as Vehicle K\", output type=\"same_entry_as\".\n"
            "- If it says \"coming from the opposite direction\", output type=\"opposite_entry_of\".\n"
            "- If it mentions \"on the main road\" or \"on the side road\", set entry_road accordingly.\n"
            "- If it mentions \"onto the main road\" or \"onto the side road\", set exit_road accordingly.\n"
            "- Only include constraints that are explicitly stated.\n"
            "\n"
            "DESCRIPTION:\n"
        )
        with open(_constraints_path, "w", encoding="utf-8") as _f:
            _f.write(_constraints_instructions)
        print(f"[INFO] Constraint-extraction prompt template saved to: {_constraints_path}")
    except Exception as _e:
        print(f"[WARNING] Failed to write constraints_prompt.txt: {_e}")


def save_aggregated_signatures_json(
    out_path: str,
    crop,
    nodes_path: str,
    params: Dict[str, Any],
    paths_named: List[Dict[str, Any]],
    lane_counts_by_road: Optional[Dict[int, int]] = None,
) -> None:
    """Optional: pure JSON version (no instructions), good for programmatic prompting."""
    payload: Dict[str, Any] = {
        "nodes": nodes_path,
        "crop_region": {"xmin": crop.xmin, "xmax": crop.xmax, "ymin": crop.ymin, "ymax": crop.ymax},
        "parameters": params,
        "num_candidates": len(paths_named),
        "candidates": paths_named,
    }
    if lane_counts_by_road:
        payload["lane_counts_by_road"] = lane_counts_by_road
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Aggregated signatures JSON saved: {out_path}")
