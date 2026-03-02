#!/usr/bin/env python3
"""Compatibility shim for schema dashboard generation.

Canonical location:
  scenario_generator/schema_dashboard.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TARGET_PATH = Path(__file__).resolve().parents[1] / "scenario_generator" / "schema_dashboard.py"
_SPEC = importlib.util.spec_from_file_location("_scenario_schema_dashboard", str(_TARGET_PATH))
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load schema dashboard module at {_TARGET_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

if hasattr(_MODULE, "__all__") and isinstance(_MODULE.__all__, list):
    names = list(_MODULE.__all__)
else:
    names = [n for n in _MODULE.__dict__.keys() if not n.startswith("_")]

for _name in names:
    globals()[_name] = getattr(_MODULE, _name)

__all__ = names

if __name__ == "__main__" and hasattr(_MODULE, "main"):
    _MODULE.main()
