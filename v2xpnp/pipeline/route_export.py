#!/usr/bin/env python3
"""Canonical route export namespace assembled from stage modules.

This preserves legacy symbol compatibility (including underscore-prefixed
helpers) while keeping implementation split into clearer, shorter files.
"""

from __future__ import annotations

from v2xpnp.pipeline import route_export_stage_01_foundation as _s1
from v2xpnp.pipeline import route_export_stage_02_alignment as _s2
from v2xpnp.pipeline import route_export_stage_03_generation as _s3
from v2xpnp.pipeline import route_export_stage_04_orchestration as _s4

for _mod in (_s1, _s2, _s3, _s4):
    for _name, _value in vars(_mod).items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value


if __name__ == "__main__":
    main()
