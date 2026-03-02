"""Stage index for the reorganized pipeline."""

from . import stage_00_cli_config
from . import stage_01_input_loading
from . import stage_02_alignment
from . import stage_03_carla_projection
from . import stage_04_postprocess
from . import stage_05_output_rendering

__all__ = [
    "stage_00_cli_config",
    "stage_01_input_loading",
    "stage_02_alignment",
    "stage_03_carla_projection",
    "stage_04_postprocess",
    "stage_05_output_rendering",
]

