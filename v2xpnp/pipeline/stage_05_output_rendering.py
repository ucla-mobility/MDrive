"""Stage 05: dataset assembly and HTML rendering/export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from v2xpnp.pipeline import pipeline_runtime


def build_dataset_payload(**kwargs: object) -> Dict[str, object]:
    """Delegate to the canonical dataset builder with unchanged behavior."""
    return pipeline_runtime.build_dataset(**kwargs)


def render_html(dataset: Dict[str, object], *, multi_mode: bool) -> str:
    return pipeline_runtime._build_html(dataset, multi_mode=bool(multi_mode))


def write_html(html: str, out_path: Path) -> Path:
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out

