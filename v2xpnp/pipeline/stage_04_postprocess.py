"""Stage 04: post-processing (overlap reduction and dedup)."""

from __future__ import annotations

from typing import Dict

from v2xpnp.pipeline import pipeline_runtime


def reduce_overlap_and_dedup(dataset: Dict[str, object], *, verbose: bool = False) -> Dict[str, object]:
    """Run the same overlap/dedup pipeline used by the runtime implementation."""
    return pipeline_runtime._apply_overlap_dedup_pipeline(dataset, verbose=bool(verbose))

