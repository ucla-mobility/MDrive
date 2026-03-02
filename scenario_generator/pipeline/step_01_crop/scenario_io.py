import json
import os
from typing import List

from .models import Scenario


def _read_txt_scenarios(path: str, source: str) -> List[Scenario]:
    lines = [ln.strip() for ln in open(path, "r", encoding="utf-8").read().splitlines()]
    items = [ln for ln in lines if ln]
    out: List[Scenario] = []
    base = os.path.splitext(os.path.basename(path))[0]
    for i, txt in enumerate(items, start=1):
        sid = f"{base}_{i:03d}"
        out.append(Scenario(sid=sid, text=txt, source=source))
    return out


def _read_json_scenarios(path: str, source: str) -> List[Scenario]:
    obj = json.load(open(path, "r", encoding="utf-8"))

    # Check if this is the new categorized format (dict with category keys and scenario lists as values)
    if isinstance(obj, dict):
        # Check if it has "scenarios" key (old format wrapper)
        if "scenarios" in obj:
            obj = obj["scenarios"]
            if not isinstance(obj, list):
                raise ValueError(f"{path}: 'scenarios' key must contain a list")
        else:
            # New categorized format: keys are categories, values are lists of scenarios
            # Check if all values are lists (categorized format) vs dict with id/text (old item format)
            has_list_values = any(isinstance(v, list) for v in obj.values())
            has_text_key = "text" in obj

            if has_list_values and not has_text_key:
                # This is the new categorized format
                out: List[Scenario] = []
                global_index = 1

                # Process categories in JSON order
                for category, scenarios in obj.items():
                    if not isinstance(scenarios, list):
                        raise ValueError(f"{path}: category '{category}' must contain a list of scenarios")

                    for scenario_text in scenarios:
                        if isinstance(scenario_text, str):
                            txt = scenario_text.strip()
                            if not txt:
                                continue
                            sid = f"{category}_{global_index:03d}"
                            out.append(Scenario(sid=sid, text=txt, source=source))
                            global_index += 1
                        elif isinstance(scenario_text, dict):
                            txt = str(scenario_text.get("text") or scenario_text.get("description") or "").strip()
                            if not txt:
                                continue
                            sid_default = f"{category}_{global_index:03d}"
                            sid = str(scenario_text.get("id", "")).strip() or sid_default
                            out.append(Scenario(sid=sid, text=txt, source=source))
                            global_index += 1

                return out
            else:
                # Old format: single dict item with id/text keys
                # Fall through to list processing below
                obj = [obj]

    # Old list format (or converted single dict)
    if not isinstance(obj, list):
        raise ValueError(f"{path}: expected list, categorized dict, or {{'scenarios':[...]}}")

    out: List[Scenario] = []
    base = os.path.splitext(os.path.basename(path))[0]
    for i, item in enumerate(obj, start=1):
        if isinstance(item, str):
            out.append(Scenario(sid=f"{base}_{i:03d}", text=item, source=source))
        elif isinstance(item, dict):
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            sid = str(item.get("id", "")).strip() or f"{base}_{i:03d}"
            out.append(Scenario(sid=sid, text=txt, source=source))
    return out


def _read_jsonl_scenarios(path: str, source: str) -> List[Scenario]:
    out: List[Scenario] = []
    base = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        i = 0
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            i += 1
            item = json.loads(ln)
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            sid = str(item.get("id", "")).strip() or f"{base}_{i:03d}"
            out.append(Scenario(sid=sid, text=txt, source=source))
    return out


def load_scenarios(scenarios_file: str, scenarios_dir: str) -> List[Scenario]:
    out: List[Scenario] = []
    if scenarios_file:
        lfn = scenarios_file.lower()
        if lfn.endswith(".txt"):
            out.extend(_read_txt_scenarios(scenarios_file, source=scenarios_file))
        elif lfn.endswith(".jsonl"):
            out.extend(_read_jsonl_scenarios(scenarios_file, source=scenarios_file))
        elif lfn.endswith(".json"):
            out.extend(_read_json_scenarios(scenarios_file, source=scenarios_file))
        else:
            out.extend(_read_txt_scenarios(scenarios_file, source=scenarios_file))
    if scenarios_dir:
        for fn in sorted(os.listdir(scenarios_dir)):
            p = os.path.join(scenarios_dir, fn)
            if not os.path.isfile(p):
                continue
            lfn = fn.lower()
            if lfn.endswith(".txt"):
                out.extend(_read_txt_scenarios(p, source=p))
            elif lfn.endswith(".jsonl"):
                out.extend(_read_jsonl_scenarios(p, source=p))
            elif lfn.endswith(".json"):
                out.extend(_read_json_scenarios(p, source=p))

    seen = set()
    uniq: List[Scenario] = []
    for s in out:
        if s.sid in seen:
            continue
        seen.add(s.sid)
        uniq.append(s)
    return uniq
