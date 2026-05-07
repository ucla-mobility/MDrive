"""Single source of truth for the CoLMDriver decision vocabulary.

Mirrors `colmdriver_action.py:411-412` (and the duplicate at :914-915). Both
the dashboard backend and frontend import these so a typo can't desynchronise
them.

Verified against `simulation/leaderboard/team_code/colmdriver_action.py` on
2026-05-01.
"""

from __future__ import annotations

# Speed (longitudinal) commands — order matches `self.speed_inten_list`.
SPEED = ("STOP", "SLOWER", "KEEP", "FASTER")

# Navigation (lateral) commands — order matches `self.nav_int_list` /
# `self.dir_inten_list`.
NAV = (
    "follow the lane",
    "left lane change",
    "right lane change",
    "go straight at intersection",
    "turn left at intersection",
    "turn right at intersection",
)

# Documented default fallback when a HITL submission times out. Logged with
# a warning, not silent.
DEFAULT_FALLBACK_SPEED = "KEEP"
DEFAULT_FALLBACK_NAV = "follow the lane"


def is_valid_speed(value: str) -> bool:
    return value in SPEED


def is_valid_nav(value: str) -> bool:
    return value in NAV


def validate_decision(decision: dict) -> None:
    """Raise ValueError if a per-ego decision dict is malformed."""
    if not isinstance(decision, dict):
        raise ValueError(f"decision must be a dict, got {type(decision).__name__}")
    speed = decision.get("speed")
    nav = decision.get("nav")
    if not is_valid_speed(speed):
        raise ValueError(f"invalid speed {speed!r}; expected one of {SPEED}")
    if not is_valid_nav(nav):
        raise ValueError(f"invalid nav {nav!r}; expected one of {NAV}")
