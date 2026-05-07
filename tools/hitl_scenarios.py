"""Scenario discovery helpers for HITL experiments.

A "scenario" is a directory with `actors_manifest.json` plus per-vehicle
route XMLs — the same shape `tools/run_custom_eval.py --routes-dir`
expects. We scan `scenarioset/interaction/<Category>/<N>/` by default.

Public API:
    list_scenarios(roots) -> List[Scenario]
    resolve_scenarios(spec, roots) -> List[Scenario]   # accepts patterns,
                                                       # categories, paths
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


_DEFAULT_ROOTS = ("scenarioset/interaction",)


@dataclass(frozen=True)
class Scenario:
    """One runnable scenario directory."""
    path: Path                 # absolute, e.g. .../Highway_On-Ramp_Merge/1
    category: str              # e.g. "Highway_On-Ramp_Merge"
    instance: str              # e.g. "1"
    ego_count: int             # number of ego entries in actors_manifest.json
    town: Optional[str]        # e.g. "Town06" (from manifest; None if absent)

    @property
    def scenario_id(self) -> str:
        return f"{self.category}_{self.instance}"

    @property
    def display(self) -> str:
        return f"{self.category}/{self.instance}"


def _scenario_from_dir(d: Path) -> Optional[Scenario]:
    manifest = d / "actors_manifest.json"
    if not manifest.is_file():
        return None
    # Don't import json eagerly at top — keeps this importable from py3.7.
    import json
    try:
        data = json.loads(manifest.read_text())
    except Exception:
        return None
    egos = data.get("ego") or []
    if not egos:
        return None
    parent = d.parent
    return Scenario(
        path=d.resolve(),
        category=parent.name,
        instance=d.name,
        ego_count=len(egos),
        town=(egos[0].get("town") if isinstance(egos[0], dict) else None),
    )


def list_scenarios(
    roots: Optional[Sequence[Path]] = None,
    repo_root: Optional[Path] = None,
) -> List[Scenario]:
    """Walk each root for `<category>/<instance>` dirs containing
    actors_manifest.json. Returns scenarios sorted by (category, instance)
    with instance-numerics natural-sorted (so "10" comes after "9")."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    roots_to_scan: List[Path] = []
    for r in roots or _DEFAULT_ROOTS:
        rp = Path(r)
        if not rp.is_absolute():
            rp = (repo_root / rp).resolve()
        if rp.is_dir():
            roots_to_scan.append(rp)

    found: List[Scenario] = []
    for root in roots_to_scan:
        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for inst_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                s = _scenario_from_dir(inst_dir)
                if s is not None:
                    found.append(s)

    def _natkey(s: Scenario):
        try:
            return (s.category, int(s.instance))
        except ValueError:
            return (s.category, 1 << 30, s.instance)

    return sorted(found, key=_natkey)


def resolve_scenarios(
    specs: Iterable[str],
    roots: Optional[Sequence[Path]] = None,
    repo_root: Optional[Path] = None,
) -> List[Scenario]:
    """Turn user inputs into a concrete list of scenarios.

    Each `spec` is one of:
      * "Highway_On-Ramp_Merge"        — category: every instance under it
      * "Highway_On-Ramp_Merge/1"      — single
      * "Highway_On-Ramp_Merge/*"      — glob over instances
      * "scenarioset/interaction/.../1"     — explicit relative or absolute path
      * "all"                          — every discovered scenario
      * shell glob e.g. "*/Highway*"   — passed through to `glob.glob`

    Order is preserved; duplicates are dropped (first occurrence wins).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    catalog = list_scenarios(roots=roots, repo_root=repo_root)
    by_path = {s.path: s for s in catalog}
    by_cat: dict = {}
    by_id: dict = {}
    for s in catalog:
        by_cat.setdefault(s.category, []).append(s)
        by_id[s.scenario_id] = s

    out: List[Scenario] = []
    seen: set = set()

    def _add(s: Scenario) -> None:
        if s.path not in seen:
            seen.add(s.path)
            out.append(s)

    for raw in specs:
        spec = raw.strip()
        if not spec:
            continue
        if spec == "all":
            for s in catalog:
                _add(s)
            continue
        # Direct path?
        path_candidate = Path(spec)
        if not path_candidate.is_absolute():
            path_candidate = (repo_root / path_candidate).resolve()
        if path_candidate.is_dir():
            sc = _scenario_from_dir(path_candidate)
            if sc is not None:
                _add(sc)
                continue
        # Category alone?
        if spec in by_cat:
            for s in by_cat[spec]:
                _add(s)
            continue
        # Category/instance shorthand?
        if "/" in spec:
            cat, _, inst = spec.partition("/")
            if cat in by_cat and inst in {"*", ""}:
                for s in by_cat[cat]:
                    _add(s)
                continue
            sid = f"{cat}_{inst}"
            if sid in by_id:
                _add(by_id[sid])
                continue
        # Shell glob fallback (resolved against repo root).
        glob_pattern = spec
        if not os.path.isabs(glob_pattern):
            glob_pattern = str(repo_root / glob_pattern)
        for match in sorted(glob.glob(glob_pattern)):
            mp = Path(match)
            if mp.is_dir():
                sc = _scenario_from_dir(mp.resolve())
                if sc is not None:
                    _add(sc)
        # If nothing matched, we silently skip — caller can detect via length.

    return out


def _format_table(scenarios: Sequence[Scenario]) -> str:
    if not scenarios:
        return "(no scenarios found)"
    # Group by category for printing.
    out_lines = []
    cur_cat = None
    for s in scenarios:
        if s.category != cur_cat:
            cur_cat = s.category
            out_lines.append(f"\n{cur_cat}:")
        town = f" [{s.town}]" if s.town else ""
        out_lines.append(f"  {s.instance:>3}  ego_count={s.ego_count}{town}")
    return "\n".join(out_lines)


def main(argv=None) -> int:
    """Quick CLI: `python -m tools.hitl_scenarios` lists everything."""
    import argparse
    p = argparse.ArgumentParser(prog="hitl_scenarios")
    p.add_argument("specs", nargs="*", default=["all"],
                   help="Scenarios to resolve (default: all)")
    p.add_argument("--root", action="append", default=None,
                   help="Override scenarioset roots (repeatable)")
    args = p.parse_args(argv)
    scenarios = resolve_scenarios(args.specs, roots=args.root)
    print(f"Found {len(scenarios)} scenario(s):")
    print(_format_table(scenarios))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
