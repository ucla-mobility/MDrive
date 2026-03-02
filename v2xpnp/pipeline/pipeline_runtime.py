#!/usr/bin/env python3
"""Canonical runtime namespace assembled from stage-oriented modules.

This preserves legacy symbol-level compatibility (including underscore helpers)
while keeping implementation split into cleaner stage files.
"""

from __future__ import annotations

from v2xpnp.pipeline import runtime_common as _s1
from v2xpnp.pipeline import runtime_projection as _s2
from v2xpnp.pipeline import runtime_postprocess as _s3
from v2xpnp.pipeline import runtime_orchestration as _s4

for _mod in (_s1, _s2, _s3, _s4):
    for _name, _value in vars(_mod).items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value


def __getattr__(name: str):
    """Resolve late-bound symbols from stage modules.

    This guards against import-order/cycle timing where a symbol is not present
    during eager namespace assembly but exists once modules finish loading.
    """
    for _mod in (_s4, _s3, _s2, _s1):
        if hasattr(_mod, name):
            return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
