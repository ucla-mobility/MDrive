"""Optional utility imports for team_code."""

try:
    import carla_birdeye_view  # noqa: F401
except Exception:
    # Optional dependency; many workflows do not install it.
    carla_birdeye_view = None
