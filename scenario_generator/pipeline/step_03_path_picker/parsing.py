import json
import re
from typing import Any, Dict, List, Optional


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract any top-level JSON object from arbitrary text using a balanced-brace scan.
    Tries full-text JSON first; then scans each brace-balanced snippet; returns the
    first one that parses and contains a top-level 'vehicles'.
    """
    # Fast path: direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "vehicles" in obj:
            return obj
    except Exception:
        pass

    start_search = 0
    n = len(text)
    while True:
        start = text.find("{", start_search)
        if start < 0:
            break
        depth = 0
        in_str = False
        esc = False
        i = start
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            else:
                if ch == '"':
                    in_str = True
                    i += 1
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        snippet = text[start:i + 1]
                        try:
                            obj = json.loads(snippet)
                            if isinstance(obj, dict) and "vehicles" in obj:
                                return obj
                        except Exception:
                            break
                i += 1
        start_search = start + 1
    return None


def _extract_json_from_codeblocks(text: str) -> Optional[Dict[str, Any]]:
    """Try parsing JSON from ```json ... ``` blocks."""
    for m in re.finditer(r"```(?:json)?\s*\n([\s\S]*?)```", text):
        block = m.group(1).strip()
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "vehicles" in obj:
                return obj
        except Exception:
            continue
    return None


def _extract_vehicles_loose(text: str) -> Optional[Dict[str, Any]]:
    """
    Last-resort loose extraction: gather pairs of vehicle/path_name from text
    without requiring valid JSON. Also captures optional confidence if nearby.
    """
    items: List[Dict[str, Any]] = []
    # Prefer explicit "path_name": "..." pairs
    for pm in re.finditer(r"\"path_name\"\s*:\s*\"([^\"]+)\"", text):
        path_name = pm.group(1)
        # Look back a bit for vehicle and confidence
        window_start = max(0, pm.start() - 300)
        ctx = text[window_start:pm.start()]
        vm = re.search(r"\"vehicle\"\s*:\s*\"([^\"]+)\"", ctx)
        cm = re.search(r"\"confidence\"\s*:\s*([0-9]*\.?[0-9]+)", ctx)
        vehicle = vm.group(1) if vm else f"Vehicle {len(items) + 1}"
        entry: Dict[str, Any] = {"vehicle": vehicle, "path_name": path_name}
        if cm:
            try:
                entry["confidence"] = float(cm.group(1))
            except Exception:
                pass
        items.append(entry)

    # If none found, fall back to any path_### token found in order
    if not items:
        paths = [m.group(0) for m in re.finditer(r"path_\d{3}[^\s\"']*", text)]
        for i, p in enumerate(paths):
            items.append({"vehicle": f"Vehicle {i + 1}", "path_name": p})

    return {"vehicles": items} if items else None


def _safe_parse_model_output(text: str) -> Optional[Dict[str, Any]]:
    """Robust parse order: (1) balanced-brace JSON (2) codeblock JSON (3) loose regex."""
    obj = _extract_first_json_object(text)
    if obj:
        return obj
    obj = _extract_json_from_codeblocks(text)
    if obj:
        return obj
    obj = _extract_vehicles_loose(text)
    if obj:
        return obj
    return None


def _safe_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse first JSON object from text (reuses existing robust extractors)."""
    obj = _extract_first_json_object(text)
    if obj:
        return obj
    obj = _extract_json_from_codeblocks(text)
    if obj:
        return obj
    return None


def _extract_description_from_prompt(prompt: str) -> str:
    """
    Best-effort extraction of the scene payload from a combined prompt.
    Supports both natural-language descriptions and schema JSON payloads.
    """
    if not prompt:
        return ""
    if "USER SCENARIO SCHEMA" in prompt:
        desc = prompt.split("USER SCENARIO SCHEMA", 1)[1].strip(": \n")
    elif "USER SCENARIO DESCRIPTION:" in prompt:
        desc = prompt.split("USER SCENARIO DESCRIPTION:", 1)[1].strip()
    else:
        desc = prompt.strip()

    desc = re.sub(r"\(Only assign paths to moving vehicles;.*?\)\s*$", "", desc, flags=re.S).strip()
    return desc
