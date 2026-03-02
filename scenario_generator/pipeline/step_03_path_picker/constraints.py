import re
from typing import Any, Dict, List, Optional, Tuple

from .parsing import _extract_description_from_prompt

# Canonical, unambiguous constraint names used everywhere (prompt + IR + solver)
_ALLOWED_CONSTRAINT_TYPES = {
    "same_approach_as",         # same ENTRY direction (approach direction)
    "opposite_approach_of",     # opposite ENTRY direction (oncoming)
    "perpendicular_right_of",   # entry is 90 degrees clockwise from reference vehicle's entry
    "perpendicular_left_of",    # entry is 90 degrees counter-clockwise from reference vehicle's entry
    "same_exit_as",             # same EXIT direction
    "same_road_as",             # same PHYSICAL road_id after the intersection (direction may differ)
    "follow_route_of",         # follower should take same route (entry + maneuver + exit corridor)
    "left_lane_of",             # adjacent lane immediately to the left (on approach)
    "right_lane_of",            # adjacent lane immediately to the right (on approach)
    "merges_into_lane_of",      # lane change: starts in adjacent lane, ends in same lane as reference
}

_OPPOSITE_CARDINAL = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _opposite_cardinal(c: str) -> str:
    return _OPPOSITE_CARDINAL.get(str(c).strip().upper(), "unknown")


def _detect_npc_vehicles(description: str) -> set:
    """
    Detect vehicles that are explicitly marked as NPC vehicles in the description.
    Returns a set of vehicle names that should NOT be treated as ego vehicles.

    Patterns detected:
    - "Vehicle X is an NPC vehicle"
    - "Vehicle X is an NPC"
    - "Vehicle X is a non-player vehicle"
    """
    npc_vehicles = set()
    desc_lower = description.lower()

    # Pattern: "Vehicle N is an NPC" or "Vehicle N is an NPC vehicle"
    pattern = r"vehicle\s+(\d+)\s+is\s+(?:an?\s+)?(?:npc|non-?player)"
    for match in re.finditer(pattern, desc_lower):
        vehicle_name = f"Vehicle {match.group(1)}"
        npc_vehicles.add(vehicle_name)

    return npc_vehicles


def _infer_constraints_from_description(
    description: str,
    norm: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add obvious inferred constraints (currently: opposite_approach_of)
    based on approach_direction fields in the normalized vehicles list.
    """
    vehicles = [v for v in norm.get("vehicles", []) if isinstance(v, dict)]
    cardinals: Dict[str, str] = {}
    for v in vehicles:
        name = v.get("vehicle")
        raw = str(v.get("approach_direction", "")).strip().upper()
        if raw in ("N", "S", "E", "W"):
            cardinals[name] = raw
    if not cardinals:
        return norm

    vehicle_names = [v.get("vehicle") for v in vehicles if v.get("vehicle")]
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    constraints = norm.get("constraints", [])

    existing_pairs = {
        (c.get("type"), c.get("a"), c.get("b"))
        for c in constraints
        if isinstance(c, dict)
    }

    def _has_opposite(a: str, b: str) -> bool:
        return (
            ("opposite_approach_of", a, b) in existing_pairs
            or ("opposite_approach_of", b, a) in existing_pairs
        )

    for i, a in enumerate(vehicle_names):
        for b in vehicle_names[i + 1:]:
            ca = cardinals.get(a)
            cb = cardinals.get(b)
            if not ca or not cb:
                continue
            if opposite.get(ca) == cb and not _has_opposite(a, b):
                constraints.append(
                    {
                        "type": "opposite_approach_of",
                        "a": a,
                        "b": b,
                        "evidence": "inferred from cardinal directions",
                    }
                )

    # Infer unary adjacent-lane requirements from phrases like
    # "in the lane to the right of Vehicle 1".
    desc_l = (description or "").lower()
    for v in vehicles:
        name = str(v.get("vehicle", "")).strip()
        if not name:
            continue
        name_l = name.lower()
        # Only match explicit "lane to/on the left/right of Vehicle X" phrases.
        right_pat = rf"(?:adjacent\\s+)?lane\\s+(?:to|on)\\s+the\\s+right\\s+of\\s+{re.escape(name_l)}"
        left_pat = rf"(?:adjacent\\s+)?lane\\s+(?:to|on)\\s+the\\s+left\\s+of\\s+{re.escape(name_l)}"
        if re.search(right_pat, desc_l):
            v["adjacent_lane_req"] = "right"
        elif re.search(left_pat, desc_l):
            v["adjacent_lane_req"] = "left"

    norm["constraints"] = constraints
    return norm


def _build_constraints_prompt(description: str) -> str:
    """
    Prompt the LLM to output ONLY constraints (no path names).
    Schema is intentionally small + aligned with natural language.
    """
    return (
        "You will read a short driving scene description.\n"
        "Extract constraints about EGO vehicles only (Vehicle 1, Vehicle 2, ...).\n"
        "Do not extract constraints about static objects such as parked vehicles, pedestrians, bicyclists, or any other prop.\n"
        "Do NOT choose path names here; only list constraints.\n"
        "\n"
        "IMPORTANT VEHICLE CLASSIFICATION:\n"
        "- EGO VEHICLES: Vehicles mentioned by name (Vehicle 1, Vehicle 2, etc.) that are NOT explicitly\n"
        "  described as \"NPC vehicle\", \"NPC\", or \"non-player character\". These need paths.\n"
        "- NPC VEHICLES: If the description says \"Vehicle X is an NPC vehicle\" or similar,\n"
        "  DO NOT include that vehicle in the 'vehicles' list. NPCs are placed as actors, not ego paths.\n"
        "  Still include constraints involving NPC vehicles if they help define other ego vehicle paths.\n"
        "\n"
        "KEY DEFINITIONS (use these exactly):\n"
        "- approach direction = the direction the vehicle APPROACHES the intersection from (ENTRY).\n"
        "- road role = whether a road is described as the \"main road\" or \"side road\".\n"
        "- lane relation = whether one ego vehicle starts in the lane to the left/right of another ego vehicle on approach.\n"
        "- same_road_as means the vehicles end up on the same PHYSICAL road (same road_id),\n"
        "  even if they travel opposite directions.\n"
        "\n"
        "Return JSON ONLY with this schema:\n"
        "{\n"
        "  \"vehicles\": [\n"
        "    {\n"
        "      \"vehicle\": \"Vehicle 1\",\n"
        "      \"maneuver\": \"left\" | \"right\" | \"straight\" | \"lane_change\" | \"unknown\",\n"
        "      \"lane_change_phase\": \"before_intersection\" | \"after_intersection\" | \"unknown\",\n"
        "      \"approach_direction\": \"N\" | \"S\" | \"E\" | \"W\" | \"unknown\",\n"
        "      \"entry_road\": \"main\" | \"side\" | \"unknown\",\n"
        "      \"exit_road\": \"main\" | \"side\" | \"unknown\"\n"
        "    }\n"
        "  ],\n"
        "  \"constraints\": [\n"
        "    {\n"
        "      \"type\": \"same_approach_as\" | \"opposite_approach_of\" | \"perpendicular_right_of\" | \"perpendicular_left_of\" | \"same_exit_as\" | \"same_road_as\" | \"follow_route_of\" | \"left_lane_of\" | \"right_lane_of\" | \"merges_into_lane_of\",\n"
        "      \"a\": \"Vehicle X\",\n"
        "      \"b\": \"Vehicle Y\",\n"
        "      \"evidence\": \"<COPIED VERBATIM from DESCRIPTION (no paraphrase)>\">\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "CRITICAL RULES (MUST FOLLOW):\n"
        "- Only include EGO vehicles in the 'vehicles' list. NPC vehicles are NOT ego vehicles.\n"
        "- Only extract constraints that are EXPLICITLY stated in the text.\n"
        "- evidence MUST be an EXACT substring copied from DESCRIPTION (no paraphrase, no synonyms).\n- keep evidence SHORT (<= 12 words).\n- keep the JSON compact: include at most 10 constraints total; omit low-confidence extras.\n"
        "- Do NOT output duplicate constraints: each (type,a,b) may appear at most once.\n"
        "- Do NOT infer entry_road or exit_road from the maneuver alone.\n"
        "- If the text does not literally say \"main road\" or \"side road\" for that vehicle, use \"unknown\".\n"
        "- If the text explicitly says \"on-ramp\", set entry_road=\"side\" for that vehicle.\n"
        "- Do NOT infer same_road_as from maneuvers. Only emit same_road_as if the description explicitly says\n"
        "  they are on the same road or one turns onto the road of the other.\n"
        "- Constraints without clear evidence will be discarded, so when unsure, omit it.\n"
        "- If a vehicle CHANGES LANES, set maneuver=\"lane_change\".\n"
        "- If the lane change happens \"after the intersection\" or \"on the exit road\", set lane_change_phase=\"after_intersection\".\n"
        "- If the lane change happens \"before the intersection\" or \"on approach\", set lane_change_phase=\"before_intersection\".\n"
        "- If lane change timing is not specified, use lane_change_phase=\"unknown\".\n"
        "- approach_direction MUST be the direction of travel (northbound/southbound/eastbound/westbound),\n"
        "  not the geographic origin. If the text says \"approaches from the north heading south\",\n"
        "  the approach_direction is \"S\" (southbound).\n"
        "- Only set approach_direction when the text explicitly states a heading or bound (north/south/east/west).\n"
        "  If it's not explicit, use \"unknown\".\n"
        "\n"
        "DISAMBIGUATION RULES (MUST FOLLOW):\n"
        "- Phrases like \"following\", \"behind\", \"tailing\" refer to a ROUTE-FOLLOWING relation (spawn/route coupling).\n"
        "  => use type=\"follow_route_of\".\n"
        "- Phrases like \"going the same direction as\" refer to APPROACH direction only.\n"
        "  => use type=\"same_approach_as\".\n"
        "- Phrases like \"coming from the opposite direction\" refer to APPROACH direction.\n"
        "  => use type=\"opposite_approach_of\".\n"
        "- Phrases like \"approaches from the perpendicular road to the right of Vehicle A's approach\"\n"
        "  => use type=\"perpendicular_right_of\". This means Vehicle a's entry is 90 degrees clockwise\n"
        "     from Vehicle b's entry (e.g., if b is westbound, a is southbound).\n"
        "- Phrases like \"approaches from the perpendicular road to the left of Vehicle A's approach\"\n"
        "  => use type=\"perpendicular_left_of\". This means Vehicle a's entry is 90 degrees counter-clockwise\n"
        "     from Vehicle b's entry (e.g., if b is westbound, a is northbound).\n"
        "- Phrases like \"turns onto Vehicle A's exit road\" or \"onto the road Vehicle A exits on\"\n"
        "  => use type=\"same_exit_as\". This means both vehicles have the same EXIT direction.\n"
        "- Phrases like \"turns onto the road Vehicle A is traveling on\" refer to PHYSICAL ROAD identity.\n"
        "  => use type=\"same_road_as\" (do NOT guess direction).\n"
        "- Phrases like \"in the lane to the left/right of Vehicle A\" refer to a LANE relation on approach.\n"
        "  => use type=\"left_lane_of\" or type=\"right_lane_of\". Do NOT treat this as road role.\n"
        "- Phrases like \"changes lanes into Vehicle A's lane\" or \"merges into Vehicle A's lane\"\n"
        "  => use type=\"merges_into_lane_of\". This means Vehicle a STARTS in an adjacent lane but\n"
        "     ENDS in the same lane as Vehicle b (lane change/merge scenario).\n"
        "\n"
        "SAFE PROPAGATION (ALLOWED INFERENCE):\n"
        "- If Vehicle A has entry_road explicitly set (main/side) AND the text says Vehicle B is\n"
        "  going the same direction as / follow_route_of Vehicle A, then set Vehicle B entry_road\n"
        "  to match Vehicle A.\n"
        "- If Vehicle A has exit_road explicitly set (main/side) AND the text says Vehicle B turns onto\n"
        "  the road Vehicle A is traveling on, then set Vehicle B exit_road to match Vehicle A.\n"
        "- Still do NOT infer road roles from left/right/straight alone.\n"
        "\n"
        "GUIDANCE:\n"
        "- Prefer binary constraints when the text is relational.\n"
        "- Prefer \"unknown\" over guessing.\n"
        "\n"
        f"DESCRIPTION:\n{description}\n"
    )


def _norm_ws(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()


def _sanitize_constraints_obj(
    constraints_obj: Dict[str, Any],
    description: str,
    max_constraints: int = 8,
) -> Dict[str, Any]:
    """
    Deterministically drop hallucinated/invalid constraints:
    - evidence must be a literal substring of description (after whitespace-normalized lowercase compare)
    - a/b must refer to vehicles in vehicles[]
    - type must be in allowed set
    - dedupe (type,a,b)
    - cap total constraints
    - filter out NPC vehicles from the vehicle list
    """
    if not isinstance(constraints_obj, dict):
        return constraints_obj

    # Detect NPC vehicles from description
    npc_vehicles = _detect_npc_vehicles(description)
    if npc_vehicles:
        print(f"[INFO] Detected NPC vehicles (not ego): {npc_vehicles}")

    vehicles = constraints_obj.get("vehicles", [])
    if not isinstance(vehicles, list):
        vehicles = []

    # Filter out NPC vehicles from the vehicle list
    filtered_vehicles = []
    for v in vehicles:
        if isinstance(v, dict) and isinstance(v.get("vehicle"), str):
            vname = v["vehicle"].strip()
            if vname in npc_vehicles:
                print(f"[INFO] Removing NPC vehicle from ego list: {vname}")
                continue
            filtered_vehicles.append(v)

    constraints_obj["vehicles"] = filtered_vehicles

    valid_names = set()
    for v in filtered_vehicles:
        if isinstance(v, dict) and isinstance(v.get("vehicle"), str) and v["vehicle"].strip():
            valid_names.add(v["vehicle"].strip())

    desc_norm = _norm_ws(description)

    raw_constraints = constraints_obj.get("constraints", [])
    if not isinstance(raw_constraints, list):
        raw_constraints = []

    kept: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    for c in raw_constraints:
        if not isinstance(c, dict):
            continue
        t = str(c.get("type", "")).strip().lower()
        a = str(c.get("a", "")).strip()
        b = str(c.get("b", "")).strip()
        ev = c.get("evidence", "")

        # Backward-compatible lane relation recovery:
        # Some models confuse lane relations ("lane to the left/right") with same_road_as.
        # If evidence explicitly mentions a lane relation, rewrite to left/right_lane_of.
        # Convention: left_lane_of/right_lane_of are asymmetric, meaning:
        #   left_lane_of:  a is in the lane to the LEFT of b (on approach)
        #   right_lane_of: a is in the lane to the RIGHT of b (on approach)
        if isinstance(ev, str) and ev.strip():
            ev_l = ev.lower()
            ev_norm_tmp = _norm_ws(ev_l)
            if "lane" in ev_norm_tmp:
                if "left" in ev_norm_tmp:
                    t = "left_lane_of"
                elif "right" in ev_norm_tmp:
                    t = "right_lane_of"

                # Try to orient (a,b) consistently with phrases like "...left of Vehicle X".
                # If we can identify a referenced vehicle in the evidence, set it as b.
                m = re.search(r"(?:to\s+the\s+)?(?:left|right)\s+of\s+(vehicle\s*\d+)", ev_l)
                if m:
                    ref = " ".join(m.group(1).split()).title()  # "Vehicle 1"
                    if ref in (a, b):
                        other_name = b if a == ref else a
                        a, b = other_name, ref

        if t not in _ALLOWED_CONSTRAINT_TYPES:
            continue
        if not a or not b or a == b:
            continue
        if a not in valid_names or b not in valid_names:
            # drops pedestrians/props/etc even if the model tried
            continue
        if not isinstance(ev, str) or not ev.strip():
            continue

        ev_norm = _norm_ws(ev)
        if ev_norm not in desc_norm:
            # evidence is not copied verbatim from description => hallucination
            continue

        key = (t, a, b)
        if key in seen:
            continue
        seen.add(key)

        kept.append({"type": t, "a": a, "b": b, "evidence": ev})
        if len(kept) >= max_constraints:
            break

    constraints_obj["constraints"] = kept
    return constraints_obj


def _normalize_constraints_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize slightly different keys into the expected {'vehicles': [...], 'constraints': [...]}."""
    if not isinstance(obj, dict):
        return None

    vehicles = obj.get("vehicles")
    if vehicles is None and isinstance(obj.get("unary"), dict):
        vehicles = []
        for v, u in obj["unary"].items():
            if not isinstance(u, dict):
                continue
            vehicles.append({
                "vehicle": v,
                "maneuver": u.get("maneuver", "unknown"),
                "entry_road": u.get("entry_road", "unknown"),
                "exit_road": u.get("exit_road", "unknown"),
            })

    constraints = obj.get("constraints")
    if constraints is None and isinstance(obj.get("binary"), list):
        constraints = obj["binary"]

    if not isinstance(vehicles, list) or not vehicles:
        return None
    if constraints is None:
        constraints = []
    if not isinstance(constraints, list):
        constraints = []

    v_out = []
    for it in vehicles:
        if not isinstance(it, dict):
            continue
        name = it.get("vehicle")
        if not isinstance(name, str) or not name.strip():
            continue
        approach_raw = str(it.get("approach_direction", "unknown")).strip().upper()
        if approach_raw in ("NORTH", "NORTHBOUND"):
            approach_raw = "N"
        elif approach_raw in ("SOUTH", "SOUTHBOUND"):
            approach_raw = "S"
        elif approach_raw in ("EAST", "EASTBOUND"):
            approach_raw = "E"
        elif approach_raw in ("WEST", "WESTBOUND"):
            approach_raw = "W"
        if approach_raw not in ("N", "S", "E", "W"):
            approach_raw = "unknown"

        v_out.append({
            "vehicle": name.strip(),
            "maneuver": str(it.get("maneuver", "unknown")).strip().lower(),
            "lane_change_phase": str(it.get("lane_change_phase", "unknown")).strip().lower(),
            "approach_direction": approach_raw,
            "entry_road": str(it.get("entry_road", "unknown")).strip().lower(),
            "exit_road": str(it.get("exit_road", "unknown")).strip().lower(),
        })

    c_out = []
    for it in constraints:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", "")).strip().lower()
        a = str(it.get("a", "")).strip()
        b = str(it.get("b", "")).strip()
        ev = str(it.get("evidence", "")).strip()
        if not t or not a or not b:
            continue
        # Keep only the small canonical set (avoids ambiguous / unsupported constraints)
        if t not in _ALLOWED_CONSTRAINT_TYPES:
            continue
        entry = {"type": t, "a": a, "b": b}
        if ev:
            entry["evidence"] = ev
        c_out.append(entry)

    return {"vehicles": v_out, "constraints": c_out}


def _filter_constraints_with_evidence(description: str, norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop constraints that lack evidence or whose evidence/trigger is not present in the description.
    Also deduplicate identical (type,a,b) tuples.
    """
    desc_l = (description or "").lower()

    def has_trigger(c: Dict[str, Any]) -> bool:
        ev = c.get("evidence", "")
        ev_l = str(ev).lower()
        if not ev_l or ev_l not in desc_l:
            return False
        t = c.get("type", "")
        if t == "same_approach_as":
            triggers = ["same direction", "following", "behind", "same approach", "same lane", "coming from the same"]
        elif t == "opposite_approach_of":
            triggers = ["opposite direction", "oncoming", "coming toward", "from the opposite"]
        elif t == "perpendicular_right_of":
            triggers = ["perpendicular", "to the right of", "right of vehicle", "cross traffic"]
        elif t == "perpendicular_left_of":
            triggers = ["perpendicular", "to the left of", "left of vehicle", "cross traffic"]
        elif t == "same_exit_as":
            triggers = ["onto vehicle", "exit road", "onto the road", "turns onto"]
        elif t == "same_road_as":
            triggers = ["same road", "onto the road", "onto vehicle", "onto the roadway", "on the road of"]
        elif t == "follow_route_of":
            triggers = ["following", "behind", "tailing", "same route", "same path"]
        elif t == "left_lane_of":
            triggers = ["lane to the left", "left lane", "in the left lane"]
        elif t == "right_lane_of":
            triggers = ["lane to the right", "right lane", "in the right lane"]
        elif t == "merges_into_lane_of":
            triggers = ["changes lanes into", "merges into", "lane change", "into vehicle", "into the lane"]
        else:
            return False
        return any(tok in ev_l or tok in desc_l for tok in triggers)

    seen = set()
    filtered = []
    for c in norm.get("constraints", []):
        t = c.get("type")
        a = c.get("a")
        b = c.get("b")
        ev = c.get("evidence", "")

        # Heuristic: if evidence says "left/right of Vehicle X" but the extracted
        # constraint points the other way, swap a/b to align with the text.
        if t in ("left_lane_of", "right_lane_of") and isinstance(ev, str):
            ev_l = ev.lower()
            m = re.search(r"(left|right)[^\n]{0,40}?\bof\s+(vehicle\s*\d+)", ev_l)
            if m:
                side = m.group(1)
                ref = " ".join(m.group(2).split()).title()  # "Vehicle 1"
                # Only swap when the side matches the constraint type and ref matches one end.
                if ((t == "left_lane_of" and side == "left") or (t == "right_lane_of" and side == "right")):
                    if ref == a and ref != b:
                        a, b = b, a
                    elif ref == b and ref != a:
                        # already consistent
                        pass

        c_use = dict(c)
        c_use["a"], c_use["b"] = a, b

        key = (c_use.get("type"), c_use.get("a"), c_use.get("b"))
        if key in seen:
            continue
        if has_trigger(c_use):
            filtered.append(c_use)
            seen.add(key)
    out = dict(norm)
    out["constraints"] = filtered
    return out


def _infer_road_role_sets(
    candidates: List[Dict[str, Any]], 
    ramp_candidate_names: Optional[set] = None
) -> Dict[str, Any]:
    """
    Infer which cardinal directions correspond to the 'main road' vs 'side road' using candidates:
    - Main road directions are the entry cardinals that have straight-through paths (entry==exit, maneuver==straight).
    - Side road directions are the remaining entry cardinals.
    - For on-ramp scenarios (when ramp_candidate_names is provided), use those to identify side entries.
    """
    entry_set = set()
    exit_set = set()
    main_cardinals = set()
    ramp_entry_cardinals = set()

    for c in candidates:
        sig = (c or {}).get("signature", {})
        ent = (sig or {}).get("entry", {})
        ex = (sig or {}).get("exit", {})
        ent_c = str((ent or {}).get("cardinal4", "")).strip().upper()
        ex_c = str((ex or {}).get("cardinal4", "")).strip().upper()
        if ent_c:
            entry_set.add(ent_c)
        if ex_c:
            exit_set.add(ex_c)

        if str((sig or {}).get("entry_to_exit_turn", "")).strip().lower() == "straight" and ent_c and ent_c == ex_c:
            main_cardinals.add(ent_c)
        
        # For on-ramp scenarios, identify ramp entry cardinals
        if ramp_candidate_names and c.get("name") in ramp_candidate_names:
            if ent_c:
                ramp_entry_cardinals.add(ent_c)

    side_entry = set([c for c in entry_set if c not in main_cardinals])

    def expand_with_opposites(s: set, universe: set) -> set:
        out = set(s)
        for c in list(s):
            oc = _opposite_cardinal(c)
            if oc in universe:
                out.add(oc)
        return out

    main_exit = expand_with_opposites(main_cardinals, exit_set)
    side_exit = expand_with_opposites(side_entry, exit_set)

    # Road-role (main/side) is only meaningful when the intersection behaves like a T-junction
    # OR when we have explicit on-ramp candidates (highway on-ramp scenarios)
    is_t_junction = (len(entry_set) == 3 and len(main_cardinals) >= 1 and len(side_entry) >= 1)
    is_on_ramp_scenario = (ramp_candidate_names is not None and len(ramp_entry_cardinals) > 0)
    
    if not is_t_junction and not is_on_ramp_scenario:
        # In 4-way (or otherwise ambiguous) intersections without on-ramps,
        # we intentionally disable this distinction.
        main_cardinals = set()
        side_entry = set()
        main_exit = set()
        side_exit = set()
    elif is_on_ramp_scenario:
        # For on-ramp scenarios, use the ramp entry cardinals as side entries
        side_entry = ramp_entry_cardinals
        side_exit = expand_with_opposites(side_entry, exit_set)
        print(f"[INFO] On-ramp role sets: main_entry={main_cardinals}, side_entry={side_entry}")

    return {
        "entry_set": entry_set,
        "exit_set": exit_set,
        "main_entry": main_cardinals,
        "side_entry": side_entry,
        "main_exit": main_exit,
        "side_exit": side_exit,
    }


def _candidate_matches_unary(
    cand: Dict[str, Any],
    maneuver: str,
    entry_road: str,
    exit_road: str,
    role_sets: Dict[str, Any],
) -> bool:
    sig = (cand or {}).get("signature", {})
    ent = (sig or {}).get("entry", {})
    ex = (sig or {}).get("exit", {})
    ent_c = str((ent or {}).get("cardinal4", "")).strip().upper()
    ex_c = str((ex or {}).get("cardinal4", "")).strip().upper()
    man = str((sig or {}).get("entry_to_exit_turn", "")).strip().lower()

    if maneuver in ("left", "right", "straight") and man != maneuver:
        return False

    # Fix A2: If the inferred role set is empty (ambiguous intersection), ignore the road-role filter.
    main_entry = role_sets.get("main_entry", set())
    side_entry = role_sets.get("side_entry", set())
    main_exit = role_sets.get("main_exit", set())
    side_exit = role_sets.get("side_exit", set())

    if entry_road == "main" and main_entry and ent_c not in main_entry:
        return False
    if entry_road == "side" and side_entry and ent_c not in side_entry:
        return False

    if exit_road == "main" and main_exit and ex_c not in main_exit:
        return False
    if exit_road == "side" and side_exit and ex_c not in side_exit:
        return False

    return True


__all__ = [
    "_ALLOWED_CONSTRAINT_TYPES",
    "_build_constraints_prompt",
    "_candidate_matches_unary",
    "_detect_npc_vehicles",
    "_extract_description_from_prompt",
    "_filter_constraints_with_evidence",
    "_infer_constraints_from_description",
    "_infer_road_role_sets",
    "_normalize_constraints_obj",
    "_opposite_cardinal",
    "_sanitize_constraints_obj",
]
