"""Stage 00: CLI/config parsing."""

from __future__ import annotations

import argparse

from v2xpnp.pipeline import pipeline_runtime


def parse_args() -> argparse.Namespace:
    return pipeline_runtime.parse_args()

