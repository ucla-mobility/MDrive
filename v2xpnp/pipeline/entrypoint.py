#!/usr/bin/env python3
"""Pipeline CLI entrypoint.

Behavior is intentionally delegated to the runtime namespace in
`pipeline_runtime.py`
to preserve output parity while allowing a cleaner package-level structure.
"""

from __future__ import annotations

from v2xpnp.pipeline import pipeline_runtime


def main() -> None:
    pipeline_runtime.main()


if __name__ == "__main__":
    main()
