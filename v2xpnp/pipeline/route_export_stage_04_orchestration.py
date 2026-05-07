"""Route export internals: export orchestration and CLI."""

from __future__ import annotations

from v2xpnp.pipeline.route_export_stage_01_foundation import *  # noqa: F401,F403
from v2xpnp.pipeline.route_export_stage_02_alignment import *  # noqa: F401,F403
from v2xpnp.pipeline.route_export_stage_03_generation import *  # noqa: F401,F403

# Explicitly import private helpers excluded by wildcard imports above
from v2xpnp.pipeline.route_export_stage_01_foundation import (  # noqa: F401
    _safe_int,
    _safe_float,
    _normalize_yaw_deg,
    _sanitize_for_json,
    _grp_yaw_diff_deg,
    _parse_lane_type_set,
    _deduplicate_cross_id_tracks,
    _augment_timing_inputs_with_ego_blockers,
    _build_actor_meta_for_timing_optimization,
    _install_carla_signal_handlers,
    _load_carla_alignment_cfg,
    _load_carla_map_cache_lines,
    _load_or_capture_carla_topdown,
    _load_vector_map,
    _merge_subdir_trajectories,
    _select_best_map,
    _transform_carla_lines,
    _apply_early_spawn_time_overrides,
    _apply_late_despawn_time_overrides,
    _maximize_safe_early_spawn_actors,
    _maximize_safe_late_despawn_actors,
)
from v2xpnp.pipeline.route_export_stage_02_alignment import (  # noqa: F401
    _build_lane_correspondence,
    _grp_align_trajectories,
    _refresh_payload_timeline_for_carla_exec,
)
from v2xpnp.pipeline.route_export_stage_03_generation import (  # noqa: F401
    _apply_lane_correspondence_to_payload,
    _build_export_payload,
    _build_html,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YAML scenarios to map-anchored replay + interactive HTML.")
    parser.add_argument("--scenario-dir", required=False, default=None, help="Scenario folder containing YAML subfolders.")
    parser.add_argument(
        "--scenario-dirs",
        nargs="+",
        default=None,
        help="Multiple scenario directories to process in batch. Each directory will be processed, simulated, aligned, and have results/videos generated. Results are named after each scenario."
    )
    parser.add_argument(
        "--subdir",
        default="all",
        help="Subfolder selector like yaml_to_carla_log (--subdir -1 / 0 / all).",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory (default: <scenario-dir>/yaml_map_export).")
    parser.add_argument(
        "--batch-results-root",
        default=None,
        help="Root directory for batch results. Each scenario gets a subdirectory named after the scenario folder."
    )
    parser.add_argument(
        "--generate-videos",
        action="store_true",
        help="Generate videos from captured images after each scenario simulation."
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for generated videos (default: 10)."
    )
    parser.add_argument(
        "--video-resize-factor",
        type=int,
        default=2,
        help="Resize factor for generated videos (default: 2). Passed to gen_video.py --resize-factor."
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Timestep spacing in seconds.")
    parser.add_argument(
        "--maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_true",
        help="Advance late-detected actors to the earliest safe spawn times (default: on).",
    )
    parser.add_argument(
        "--no-maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_false",
        help="Disable early-spawn timing optimization.",
    )
    parser.set_defaults(maximize_safe_early_spawn=True)
    parser.add_argument(
        "--early-spawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (m) for early-spawn interference checks (default: 0.25).",
    )
    parser.add_argument(
        "--maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_true",
        help="Extend actor lifetimes toward scenario horizon when safe (default: on).",
    )
    parser.add_argument(
        "--no-maximize-safe-late-despawn",
        dest="maximize_safe_late_despawn",
        action="store_false",
        help="Disable late-despawn timing optimization.",
    )
    parser.set_defaults(maximize_safe_late_despawn=True)
    parser.add_argument(
        "--late-despawn-safety-margin",
        type=float,
        default=0.25,
        help="Extra safety margin (m) for late-despawn hold checks (default: 0.25).",
    )
    parser.add_argument(
        "--timing-optimization-report",
        default=None,
        help="Optional JSON path for detailed timing optimization report.",
    )
    parser.add_argument(
        "--lane-change-filter",
        dest="lane_change_filter",
        action="store_true",
        help="Stabilize lane snapping with hysteresis to suppress noise-driven rapid lane flips (default: on).",
    )
    parser.add_argument(
        "--no-lane-change-filter",
        dest="lane_change_filter",
        action="store_false",
        help="Disable lane-change stabilization filter.",
    )
    parser.set_defaults(lane_change_filter=True)
    parser.add_argument(
        "--lane-change-confirm-window",
        type=int,
        default=5,
        help="Look-ahead window (frames) for confirming lane changes (default: 5).",
    )
    parser.add_argument(
        "--lane-change-confirm-votes",
        type=int,
        default=3,
        help="Minimum supporting observations in confirm window to accept a lane change (default: 3).",
    )
    parser.add_argument(
        "--lane-change-cooldown-frames",
        type=int,
        default=3,
        help="Minimum cooldown frames after a lane change before accepting another weak change (default: 3).",
    )
    parser.add_argument(
        "--lane-change-endpoint-guard-frames",
        type=int,
        default=4,
        help="Extra-guard frames near trajectory start/end against spurious lane changes (default: 4).",
    )
    parser.add_argument(
        "--lane-change-endpoint-extra-votes",
        type=int,
        default=1,
        help="Additional votes required for lane changes near endpoints (default: 1).",
    )
    parser.add_argument(
        "--lane-change-min-improvement-m",
        type=float,
        default=0.2,
        help="Minimum lane-distance improvement (m) that can justify a change with moderate evidence (default: 0.2).",
    )
    parser.add_argument(
        "--lane-change-keep-lane-max-dist",
        type=float,
        default=3.0,
        help="If current lane projection distance exceeds this (m), allow switching more readily (default: 3.0).",
    )
    parser.add_argument(
        "--lane-change-short-run-max",
        type=int,
        default=2,
        help="Max internal run length (frames) considered jitter and collapsed in post-filter pass (default: 2).",
    )
    parser.add_argument(
        "--lane-change-endpoint-short-run",
        type=int,
        default=2,
        help="Max start/end run length (frames) considered jitter and collapsed (default: 2).",
    )
    parser.add_argument(
        "--lane-snap-top-k",
        type=int,
        default=8,
        help="Number of candidate lanes considered per point for stabilized snapping (default: 8).",
    )
    parser.add_argument(
        "--vehicle-forbidden-lane-types",
        default="2",
        help=(
            "Comma/space-separated lane types that motor vehicles can never snap to "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--vehicle-parked-only-lane-types",
        default="3",
        help=(
            "Comma/space-separated lane types only parked motor vehicles may snap to "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--parked-net-disp-max-m",
        type=float,
        default=1.0,
        help="Parked detection threshold: max start-to-end displacement (m) (default: 1.0).",
    )
    parser.add_argument(
        "--parked-radius-p90-max-m",
        type=float,
        default=1.1,
        help="Parked detection threshold: max p90 radius around median pose (m) (default: 1.1).",
    )
    parser.add_argument(
        "--parked-radius-max-m",
        type=float,
        default=2.0,
        help="Parked detection threshold: max radius around median pose (m) (default: 2.0).",
    )
    parser.add_argument(
        "--parked-p95-step-max-m",
        type=float,
        default=0.55,
        help="Parked detection threshold: max p95 frame-to-frame step (m) (default: 0.55).",
    )
    parser.add_argument(
        "--parked-max-from-start-m",
        type=float,
        default=1.8,
        help="Parked detection threshold: max distance from first pose (m) (default: 1.8).",
    )
    parser.add_argument(
        "--parked-large-step-threshold-m",
        type=float,
        default=0.6,
        help="Parked detection: step size considered a large jump (m) (default: 0.6).",
    )
    parser.add_argument(
        "--parked-large-step-max-ratio",
        type=float,
        default=0.08,
        help="Parked detection: max fraction of large-jump steps (default: 0.08).",
    )
    parser.add_argument(
        "--parked-robust-cluster",
        dest="parked_robust_cluster",
        action="store_true",
        help="Enable robust parked detection using dominant inlier cluster (default: on).",
    )
    parser.add_argument(
        "--no-parked-robust-cluster",
        dest="parked_robust_cluster",
        action="store_false",
        help="Disable robust inlier-cluster parked detection.",
    )
    parser.set_defaults(parked_robust_cluster=True)
    parser.add_argument(
        "--parked-robust-cluster-eps-m",
        type=float,
        default=0.8,
        help="Robust parked detection: inlier cluster radius (m) (default: 0.8).",
    )
    parser.add_argument(
        "--parked-robust-min-inlier-ratio",
        type=float,
        default=0.8,
        help="Robust parked detection: minimum inlier ratio (default: 0.8).",
    )
    parser.add_argument(
        "--parked-robust-max-outlier-run",
        type=int,
        default=3,
        help="Robust parked detection: max contiguous outlier frames (default: 3).",
    )
    parser.add_argument(
        "--parked-robust-min-points",
        type=int,
        default=6,
        help="Robust parked detection: minimum trajectory points (default: 6).",
    )
    parser.add_argument(
        "--id-merge-distance-m",
        type=float,
        default=8.0,
        help=(
            "When same numeric actor id appears in multiple subdirs, trajectories farther than this "
            "distance are split into separate actors (default: 8.0). Set <=0 to preserve legacy merging."
        ),
    )
    parser.add_argument(
        "--cross-id-dedup",
        dest="cross_id_dedup",
        action="store_true",
        help="Merge near-identical tracks even if their actor IDs differ (default: on).",
    )
    parser.add_argument(
        "--no-cross-id-dedup",
        dest="cross_id_dedup",
        action="store_false",
        help="Disable cross-ID overlap deduplication.",
    )
    parser.set_defaults(cross_id_dedup=True)
    parser.add_argument(
        "--cross-id-dedup-max-median-dist-m",
        type=float,
        default=1.2,
        help="Max median XY distance for cross-ID dedup (default: 1.2m).",
    )
    parser.add_argument(
        "--cross-id-dedup-max-p90-dist-m",
        type=float,
        default=2.0,
        help="Max p90 XY distance for cross-ID dedup (default: 2.0m).",
    )
    parser.add_argument(
        "--cross-id-dedup-max-median-yaw-diff-deg",
        type=float,
        default=35.0,
        help="Max median yaw difference for non-walker cross-ID dedup (default: 35 deg).",
    )
    parser.add_argument(
        "--cross-id-dedup-min-common-points",
        type=int,
        default=8,
        help="Minimum overlapping timesteps required for cross-ID dedup (default: 8).",
    )
    parser.add_argument(
        "--cross-id-dedup-min-overlap-each",
        type=float,
        default=0.30,
        help="Minimum overlap ratio for each track in a dedup pair (default: 0.30).",
    )
    parser.add_argument(
        "--cross-id-dedup-min-overlap-any",
        type=float,
        default=0.75,
        help="Minimum overlap ratio for at least one track in a dedup pair (default: 0.75).",
    )
    parser.add_argument(
        "--map-pkl",
        action="append",
        default=[],
        help="Map pickle path (repeatable). If omitted, uses corridors + intersection defaults.",
    )
    parser.add_argument(
        "--map-selection-sample-count",
        type=int,
        default=1200,
        help="Max sampled trajectory points for selecting best map (default: 1200).",
    )
    parser.add_argument(
        "--map-selection-bbox-margin",
        type=float,
        default=20.0,
        help="BBox margin (m) for map selection outside-penalty (default: 20).",
    )
    parser.add_argument(
        "--carla-map-layer",
        dest="carla_map_layer",
        action="store_true",
        help="Include aligned CARLA map polylines as an optional HTML map layer (default: on).",
    )
    parser.add_argument(
        "--no-carla-map-layer",
        dest="carla_map_layer",
        action="store_false",
        help="Disable CARLA map layer in output payload/HTML.",
    )
    parser.add_argument(
        "--skip-map-snap-compute",
        action="store_true",
        help="Skip per-frame CARLA projection (slow step). CARLA layer and lane correspondence cache still load. 'Use map snapped poses' won't work.",
    )
    parser.set_defaults(carla_map_layer=True)
    parser.add_argument(
        "--carla-map-cache",
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_map_cache.pkl",
        help="Path to cached CARLA map polylines pickle (default: v2xpnp/map/carla_map_cache.pkl).",
    )
    parser.add_argument(
        "--carla-map-offset-json",
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="CARLA->V2XPNP alignment JSON (tx/ty/theta/flip_y/scale) for CARLA map layer.",
    )
    parser.add_argument(
        "--carla-map-image-cache",
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_topdown_cache.jpg",
        help="Path to cached CARLA top-down JPEG image (default: v2xpnp/map/carla_topdown_cache.jpg).",
    )
    parser.add_argument(
        "--carla-host",
        default="localhost",
        help="CARLA server hostname for top-down image capture (default: localhost).",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2005,
        help="CARLA server RPC port for top-down image capture (default: 2005).",
    )
    parser.add_argument(
        "--start-carla",
        default=True,
        action="store_true",
        help="Automatically launch a local CARLA server before GRP alignment/CARLA operations.",
    )
    parser.add_argument(
        "--carla-root",
        type=str,
        default=None,
        help=(
            "Path to CARLA installation directory containing CarlaUE4.sh. "
            "Defaults to CARLA_ROOT env var or ./carla912."
        ),
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Extra arguments to pass to CarlaUE4.sh (repeatable).",
    )
    parser.add_argument(
        "--carla-port-tries",
        type=int,
        default=CARLA_PORT_TRIES,
        help=f"How many ports to try if the desired CARLA port is already in use (default: {CARLA_PORT_TRIES}).",
    )
    parser.add_argument(
        "--carla-port-step",
        type=int,
        default=CARLA_PORT_STEP,
        help=f"Port increment when searching for a free CARLA port (default: {CARLA_PORT_STEP}).",
    )
    parser.add_argument(
        "--carla-map-name",
        default="ucla_v2",
        help="CARLA map name to load before capturing top-down image (default: ucla_v2).",
    )
    parser.add_argument(
        "--capture-carla-image",
        dest="capture_carla_image",
        action="store_true",
        help="Attempt to capture a top-down image from a running CARLA server if no cache exists (default: on).",
    )
    parser.add_argument(
        "--no-capture-carla-image",
        dest="capture_carla_image",
        action="store_false",
        help="Disable CARLA image capture; only use cached image.",
    )
    parser.set_defaults(capture_carla_image=True)
    parser.add_argument(
        "--carla-topdown-method",
        type=str,
        choices=["ortho", "tiled"],
        default="ortho",
        help=(
            "How to capture the CARLA top-down image. 'ortho' (default): one wide-FOV "
            "shot orthorectified using lane waypoints as 3D ground control points. "
            "'tiled': legacy multi-tile near-orthographic stitch."
        ),
    )
    parser.add_argument("--carla-topdown-altitude", type=float, default=1500.0)
    parser.add_argument("--carla-topdown-fov-deg", type=float, default=60.0)
    parser.add_argument("--carla-topdown-image-px", type=int, default=4096)
    parser.add_argument("--carla-topdown-waypoint-spacing-m", type=float, default=4.0)
    parser.add_argument("--carla-topdown-px-per-meter", type=float, default=3.0)
    parser.add_argument(
        "--lane-correspondence",
        dest="lane_correspondence",
        action="store_true",
        help="Build robust V2XPNP-lane to CARLA-line correspondence and snap actors accordingly (default: on).",
    )
    parser.add_argument(
        "--no-lane-correspondence",
        dest="lane_correspondence",
        action="store_false",
        help="Disable lane correspondence and CARLA-lane actor snapping.",
    )
    parser.set_defaults(lane_correspondence=True)
    parser.add_argument(
        "--lane-correspondence-driving-types",
        default="1",
        help="Comma/space lane types treated as driving lanes for one-to-one lane correspondence (default: 1).",
    )
    parser.add_argument(
        "--lane-correspondence-top-k",
        type=int,
        default=28,
        help="Candidate CARLA lines examined per V2 lane during correspondence (default: 28).",
    )
    parser.add_argument(
        "--lane-correspondence-cache-dir",
        type=str,
        default="__script_dir__",
        help="Directory for caching lane correspondence results (default: alongside this script). Use '__output_dir__' to stash next to scenario output.",
    )
    parser.add_argument(
        "--snap-to-map",
        dest="snap_to_map",
        action="store_true",
        help="Use map-matched coordinates as rendered/exported pose (default: on).",
    )
    parser.add_argument(
        "--no-snap-to-map",
        dest="snap_to_map",
        action="store_false",
        help="Keep raw YAML coordinates as rendered pose while still reporting matched lanes.",
    )
    parser.set_defaults(snap_to_map=True)
    parser.add_argument(
        "--map-max-points-per-line",
        type=int,
        default=600,
        help="Max points per lane polyline in HTML payload (default: 600).",
    )

    # --- GRP trajectory alignment ---
    parser.add_argument(
        "--grp-align",
        dest="grp_align",
        action="store_true",
        help="Enable GRP-aware trajectory alignment via CARLA GlobalRoutePlanner (default: off).",
    )
    parser.add_argument(
        "--no-grp-align",
        dest="grp_align",
        action="store_false",
        help="Disable GRP trajectory alignment.",
    )
    parser.set_defaults(grp_align=True)
    parser.add_argument(
        "--grp-snap-radius",
        type=float,
        default=2.5,
        help="GRP alignment: search radius for CARLA waypoint candidates (default: 2.5m).",
    )
    parser.add_argument(
        "--grp-snap-k",
        type=int,
        default=6,
        help="GRP alignment: max candidates per input waypoint (default: 6).",
    )
    parser.add_argument(
        "--grp-heading-thresh",
        type=float,
        default=40.0,
        help="GRP alignment: yaw tolerance for candidate filtering in degrees (default: 40).",
    )
    parser.add_argument(
        "--grp-lane-change-penalty",
        type=float,
        default=50.0,
        help="GRP alignment: DP cost penalty for switching lanes (default: 50).",
    )
    parser.add_argument(
        "--grp-actor-max-median-displacement-m",
        type=float,
        default=2.0,
        help=(
            "Reject aligned actor trajectories if median XY displacement from source exceeds this "
            "threshold (default: 2.0m)."
        ),
    )
    parser.add_argument(
        "--grp-actor-max-p90-displacement-m",
        type=float,
        default=4.0,
        help=(
            "Reject aligned actor trajectories if p90 XY displacement from source exceeds this "
            "threshold (default: 4.0m)."
        ),
    )
    parser.add_argument(
        "--grp-actor-max-displacement-m",
        type=float,
        default=10.0,
        help=(
            "Reject aligned actor trajectories if max XY displacement from source exceeds this "
            "threshold (default: 10.0m)."
        ),
    )
    parser.add_argument(
        "--grp-ego-max-median-displacement-m",
        type=float,
        default=1.25,
        help=(
            "Reject aligned ego trajectories if median XY displacement from source exceeds this "
            "threshold (default: 1.25m)."
        ),
    )
    parser.add_argument(
        "--grp-ego-max-p90-displacement-m",
        type=float,
        default=2.5,
        help=(
            "Reject aligned ego trajectories if p90 XY displacement from source exceeds this "
            "threshold (default: 2.5m)."
        ),
    )
    parser.add_argument(
        "--grp-ego-max-displacement-m",
        type=float,
        default=6.0,
        help=(
            "Reject aligned ego trajectories if max XY displacement from source exceeds this "
            "threshold (default: 6.0m)."
        ),
    )
    parser.add_argument(
        "--grp-sampling-resolution",
        type=float,
        default=2.0,
        help="GRP alignment: GlobalRoutePlanner sampling resolution in meters (default: 2.0).",
    )

    # --- Walker sidewalk compression and stabilization ---
    parser.add_argument(
        "--walker-sidewalk-compression",
        dest="walker_sidewalk_compression",
        action="store_true",
        help="Enable walker sidewalk compression and spawn stabilization (default: on).",
    )
    parser.add_argument(
        "--no-walker-sidewalk-compression",
        dest="walker_sidewalk_compression",
        action="store_false",
        help="Disable walker sidewalk compression.",
    )
    parser.set_defaults(walker_sidewalk_compression=True)
    parser.add_argument(
        "--walker-lane-spacing-m",
        type=float,
        default=None,
        help="Lane spacing for sidewalk geometry (default: auto-calibrate from map).",
    )
    parser.add_argument(
        "--walker-sidewalk-start-factor",
        type=float,
        default=0.5,
        help="Sidewalk start distance = factor * lane_spacing (default: 0.5).",
    )
    parser.add_argument(
        "--walker-sidewalk-outer-factor",
        type=float,
        default=3.0,
        help="Sidewalk outer band factor k: y = k * sidewalk_start_distance (default: 3.0, range 2-4).",
    )
    parser.add_argument(
        "--walker-compression-target-band-m",
        type=float,
        default=2.5,
        help="Target sidewalk width after compression in meters (default: 2.5).",
    )
    parser.add_argument(
        "--walker-compression-power",
        type=float,
        default=1.5,
        help="Nonlinear compression power (>1 = stronger compression at distance) (default: 1.5).",
    )
    parser.add_argument(
        "--walker-min-spawn-separation-m",
        type=float,
        default=0.8,
        help="Minimum separation between walker spawn positions (default: 0.8).",
    )
    parser.add_argument(
        "--walker-radius-m",
        type=float,
        default=0.35,
        help="Approximate walker collision radius (default: 0.35).",
    )
    parser.add_argument(
        "--walker-crossing-road-ratio-thresh",
        type=float,
        default=0.15,
        help="Road occupancy ratio threshold for crossing classification (default: 0.15).",
    )
    parser.add_argument(
        "--walker-crossing-lateral-thresh-m",
        type=float,
        default=4.0,
        help="Lateral traversal distance indicating crossing behavior (default: 4.0).",
    )
    parser.add_argument(
        "--walker-road-presence-min-frames",
        type=int,
        default=5,
        help="Min sustained frames in road region for crossing classification (default: 5).",
    )
    parser.add_argument(
        "--walker-max-lateral-offset-m",
        type=float,
        default=3.0,
        help="Maximum allowed lateral offset from compression (default: 3.0).",
    )
    # --- CARLA Route Export Options ---
    parser.add_argument(
        "--export-carla-routes",
        default=True,
        action="store_true",
        help="Export CARLA-compatible XML route files for use with run_custom_eval.py",
    )
    parser.add_argument(
        "--carla-routes-dir",
        type=str,
        default=None,
        help="Output directory for CARLA routes (default: <output-dir>/carla_routes/)",
    )
    parser.add_argument(
        "--carla-town",
        type=str,
        default="ucla_v2",
        help="CARLA town name for route XML files (default: ucla_v2)",
    )
    parser.add_argument(
        "--carla-route-id",
        type=str,
        default="0",
        help="Route ID to use in CARLA route XML files (default: 0)",
    )
    parser.add_argument(
        "--carla-ego-path-source",
        choices=("auto", "raw", "map", "corr"),
        default="auto",
        help=(
            "Source trajectory used when exporting ego route XML. "
            "'auto' prefers map-snapped poses, then raw. "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--carla-actor-control-mode",
        choices=("policy", "replay"),
        default="policy",
        help=(
            "Control mode for NPC vehicles. 'policy' uses CARLA's AI planners (realistic driving). "
            "'replay' uses exact logged transform replay. (default: policy)"
        ),
    )
    parser.add_argument(
        "--carla-walker-control-mode",
        choices=("policy", "replay"),
        default="policy",
        help=(
            "Control mode for walkers. 'policy' uses CARLA's walker AI. "
            "'replay' uses exact logged transform replay. (default: policy)"
        ),
    )
    parser.add_argument(
        "--carla-encode-timing",
        action="store_true",
        default=True,
        help="Include timing information in CARLA route waypoints (default: enabled)",
    )
    parser.add_argument(
        "--no-carla-encode-timing",
        dest="carla_encode_timing",
        action="store_false",
        help="Disable timing information in CARLA route waypoints",
    )
    parser.add_argument(
        "--carla-snap-to-road",
        action="store_true",
        default=False,
        help="Enable snap_to_road in CARLA route files (default: disabled for accuracy)",
    )
    parser.add_argument(
        "--carla-static-spawn-only",
        action="store_true",
        default=False,
        help="For parked vehicles, only output spawn position (no trajectory) for efficiency",
    )
    # --- Run CARLA Scenario After Export ---
    parser.add_argument(
        "--run-custom-eval",
        default=True,
        action="store_true",
        help="After exporting CARLA routes, automatically run the scenario using tools/run_custom_eval.py",
    )
    parser.add_argument(
        "--eval-planner",
        type=str,
        default="log-replay",
        help="Planner for run_custom_eval (default: 'log-replay' for exact trajectory replay).",
    )
    parser.add_argument(
        "--eval-port",
        type=int,
        default=None,
        help="CARLA port for run_custom_eval (default: same as --carla-port)",
    )
    parser.add_argument(
        "--eval-overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing evaluation results (default: enabled)",
    )
    parser.add_argument(
        "--eval-timeout-factor",
        type=float,
        default=2.0,
        help="Timeout multiplier for scenario duration (default: 2.0)",
    )
    parser.add_argument(
        "--capture-logreplay-images",
        default=True,
        action="store_true",
        help="Save camera images during log-replay evaluation (passed to run_custom_eval.py)",
    )
    parser.add_argument(
        "--capture-every-sensor-frame",
        action="store_true",
        help="Save RGB/BEV images for every sensor frame during evaluation (passed to run_custom_eval.py)",
    )
    parser.add_argument(
        "--npc-only-fake-ego",
        dest="npc_only_fake_ego",
        action="store_true",
        default=True,
        help=(
            "Run evaluation in no-ego mode: convert exported ego trajectories into custom NPC actors "
            "for this run and capture images from their perspective (default: enabled)."
        ),
    )
    parser.add_argument(
        "--real-ego-eval",
        dest="npc_only_fake_ego",
        action="store_false",
        help=(
            "Disable no-ego fake-ego mode and run with real ego vehicles "
            "(keeps planner-compatible ego route execution)."
        ),
    )
    return parser.parse_args()


# =============================================================================
# CARLA Route Export Module
# =============================================================================
# This module exports trajectories to CARLA-compatible XML route files
# that work with run_custom_eval.py and the CARLA leaderboard.
# =============================================================================

def _write_carla_route_xml(
    path: Path,
    route_id: str,
    role: str,
    town: str,
    waypoints: List[Waypoint],
    times: Optional[List[float]] = None,
    snap_to_road: bool = False,
    control_mode: str = "policy",
    target_speed_mps: Optional[float] = None,
    model: Optional[str] = None,
    speeds: Optional[List[float]] = None,
) -> None:
    """Write a CARLA-compatible route XML file with timing and control mode."""
    root = ET.Element("routes")
    route_attrs = {
        "id": str(route_id),
        "town": town,
        "role": role,
        "snap_to_road": "true" if snap_to_road else "false",
    }
    if control_mode:
        route_attrs["control_mode"] = control_mode
    if target_speed_mps is not None and target_speed_mps > 0:
        route_attrs["target_speed"] = f"{target_speed_mps:.2f}"
    if model:
        route_attrs["model"] = model
    
    route_elem = ET.SubElement(root, "route", route_attrs)
    
    for idx, wp in enumerate(waypoints):
        attrs = {
            "x": f"{float(wp.x):.6f}",
            "y": f"{float(wp.y):.6f}",
            "z": f"{float(wp.z):.6f}",
            "yaw": f"{float(wp.yaw):.6f}",
            "pitch": "0.000000",
            "roll": "0.000000",
        }
        if times is not None and idx < len(times):
            try:
                attrs["time"] = f"{float(times[idx]):.6f}"
            except (TypeError, ValueError):
                pass
        if speeds is not None and idx < len(speeds):
            try:
                attrs["speed"] = f"{float(speeds[idx]):.4f}"
            except (TypeError, ValueError):
                pass
        ET.SubElement(route_elem, "waypoint", attrs)
    
    tree = ET.ElementTree(root)
    # ET.indent() requires Python 3.9+; use fallback for older versions
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    else:
        _indent_xml(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add indentation to XML tree for pretty printing (Python <3.9 fallback)."""
    indent_str = "\n" + "  " * level
    if len(elem):  # has children
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str


def _compute_target_speed(
    waypoints: List[Waypoint],
    times: Optional[List[float]],
    default_dt: float = 0.1,
    window_seconds: float = 2.0,
    moving_threshold_mps: float = 0.3,
) -> float:
    """Compute a robust representative cruising speed from a trajectory.

    The old approach summed consecutive-point distances, which inflated speed
    dramatically due to GPS/lidar tracking noise (e.g. 18 m real displacement
    computed as 280 m cumulative path → 27 m/s instead of 1.8 m/s).

    This version uses **displacement-based sliding windows**:
      1. For each point, look ahead by *window_seconds* and compute the
         straight-line displacement divided by elapsed time.  This naturally
         cancels out high-frequency jitter.
      2. Discard windows where the vehicle is essentially stationary
         (speed < *moving_threshold_mps*).
      3. Return the **75th-percentile** of the remaining window speeds,
         which represents the vehicle's typical cruising speed while
         filtering out acceleration/deceleration transients.

    Falls back to total displacement / total duration when no moving
    windows are found (very slow vehicles).
    """
    n = len(waypoints)
    if n < 2:
        return 0.0

    # Build time array
    if times and len(times) >= n:
        t = [float(ti) for ti in times[:n]]
    else:
        t = [float(default_dt) * i for i in range(n)]

    total_dt = t[-1] - t[0]
    if total_dt < 1e-6:
        return 0.0

    # --- Sliding-window displacement speeds ---
    window_speeds: List[float] = []
    j = 0  # leading pointer
    for i in range(n):
        # Advance j so that t[j] - t[i] >= window_seconds
        while j < n - 1 and (t[j] - t[i]) < window_seconds:
            j += 1
        dt_w = t[j] - t[i]
        if dt_w < 1e-6:
            continue
        dx = float(waypoints[j].x) - float(waypoints[i].x)
        dy = float(waypoints[j].y) - float(waypoints[i].y)
        disp = math.hypot(dx, dy)
        window_speeds.append(disp / dt_w)

    # Keep only windows where the vehicle is actually moving
    moving_speeds = [s for s in window_speeds if s >= moving_threshold_mps]

    if moving_speeds:
        moving_speeds.sort()
        # 75th percentile — robust cruising speed
        idx_75 = int(len(moving_speeds) * 0.75)
        idx_75 = min(idx_75, len(moving_speeds) - 1)
        return moving_speeds[idx_75]

    # Fallback: total displacement / total time (vehicle barely moved)
    dx_total = float(waypoints[-1].x) - float(waypoints[0].x)
    dy_total = float(waypoints[-1].y) - float(waypoints[0].y)
    return math.hypot(dx_total, dy_total) / total_dt


def _sanitize_trajectory(
    waypoints: List[Waypoint],
    times: List[float],
    default_dt: float = 0.1,
    max_plausible_speed_mps: float = 40.0,
) -> Tuple[List[Waypoint], List[float]]:
    """Remove tracking-noise artefacts (zigzag / ID-swap / teleportation).

    The tracker sometimes assigns the same ID to two nearby vehicles on
    alternating frames, producing impossible ~10 m jumps at 10 Hz (≡ 100 m/s).
    This function detects such frames and replaces them by holding the last
    good position, then does a light median-filter pass to smooth residual
    jitter.

    Parameters
    ----------
    waypoints : list of Waypoint
        Raw trajectory in CARLA coordinates.
    times : list of float
        Corresponding timestamps.
    max_plausible_speed_mps : float
        Any instantaneous displacement/dt above this is treated as a glitch.
        Default 40 m/s ≈ 144 km/h — generous enough for highway traffic.

    Returns
    -------
    (cleaned_wps, cleaned_times) with the same length as the input.
    """
    n = len(waypoints)
    if n < 3:
        return list(waypoints), list(times)

    dt = default_dt

    # ---- Pass 1: flag impossible jumps ----
    xs = [float(wp.x) for wp in waypoints]
    ys = [float(wp.y) for wp in waypoints]
    yaws = [float(wp.yaw) for wp in waypoints]
    zs = [float(wp.z) for wp in waypoints]

    bad = [False] * n  # True → this frame is an artefact
    for i in range(1, n):
        dti = (times[i] - times[i - 1]) if i < len(times) else dt
        if dti < 1e-6:
            dti = dt
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = math.hypot(dx, dy)
        speed = dist / dti
        if speed > max_plausible_speed_mps:
            bad[i] = True

    # Also flag zigzag patterns: if i-1 is fine, i is bad, i+1 returns to
    # roughly the same position as i-1 → classic ID-swap.  Mark i as bad.
    for i in range(1, n - 1):
        if bad[i]:
            continue
        if not bad[i - 1] and not bad[i + 1]:
            # Check if i is a single-frame outlier
            d_prev = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            d_next = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            d_skip = math.hypot(xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1])
            # If skipping this frame produces a much shorter path → outlier
            if d_prev + d_next > 3 * (d_skip + 0.1):
                dti = (times[i] - times[i - 1]) if i < len(times) else dt
                if dti < 1e-6:
                    dti = dt
                if d_prev / dti > max_plausible_speed_mps * 0.5:
                    bad[i] = True

    n_bad = sum(bad)
    if n_bad > 0:
        # Replace bad frames by holding the last good position
        for i in range(1, n):
            if bad[i]:
                # Find the previous good frame
                j = i - 1
                while j > 0 and bad[j]:
                    j -= 1
                xs[i] = xs[j]
                ys[i] = ys[j]
                zs[i] = zs[j]
                yaws[i] = yaws[j]

    # ---- Pass 2: Lightweight median filter (window=3) on x,y ----
    # Suppresses residual 1-frame jitter without distorting turns.
    fxs = list(xs)
    fys = list(ys)
    for i in range(1, n - 1):
        vals_x = sorted([xs[i - 1], xs[i], xs[i + 1]])
        vals_y = sorted([ys[i - 1], ys[i], ys[i + 1]])
        fxs[i] = vals_x[1]
        fys[i] = vals_y[1]

    # ---- Pass 3: Remove consecutive duplicate positions (dedup) ----
    # After fixing artefacts, many consecutive frames have identical (x,y).
    # Keep only the first and last of each constant run to avoid WaypointFollower
    # getting stuck trying to reach an already-reached point.
    clean_wps: List[Waypoint] = []
    clean_times: List[float] = []
    i = 0
    while i < n:
        # Start of a run of identical positions
        j = i + 1
        while j < n and abs(fxs[j] - fxs[i]) < 0.05 and abs(fys[j] - fys[i]) < 0.05:
            j += 1
        # Keep first frame of the run
        clean_wps.append(Waypoint(x=fxs[i], y=fys[i], z=zs[i], yaw=yaws[i]))
        clean_times.append(times[i])
        # If the run spans >2 frames, also keep the last (to preserve timing)
        if j - i > 2:
            last = j - 1
            clean_wps.append(Waypoint(x=fxs[last], y=fys[last], z=zs[last], yaw=yaws[last]))
            clean_times.append(times[last])
        elif j - i == 2:
            # Keep both frames (they're not truly duplicates, just close)
            clean_wps.append(Waypoint(x=fxs[i + 1], y=fys[i + 1], z=zs[i + 1], yaw=yaws[i + 1]))
            clean_times.append(times[i + 1])
        i = j

    if len(clean_wps) < 2 and n >= 2:
        # Degenerate: entire trajectory was one position; keep first and last
        clean_wps = [Waypoint(x=fxs[0], y=fys[0], z=zs[0], yaw=yaws[0]),
                     Waypoint(x=fxs[-1], y=fys[-1], z=zs[-1], yaw=yaws[-1])]
        clean_times = [times[0], times[-1]]

    return clean_wps, clean_times


def _trajectory_instability_score(
    waypoints: Sequence[Waypoint],
    jump_threshold_m: float = 2.8,
    yaw_flip_threshold_deg: float = 120.0,
) -> float:
    """Heuristic instability score for export-track source selection."""
    if len(waypoints) < 2:
        return 0.0
    score = 0.0
    for i in range(1, len(waypoints)):
        dx = float(waypoints[i].x) - float(waypoints[i - 1].x)
        dy = float(waypoints[i].y) - float(waypoints[i - 1].y)
        if math.hypot(dx, dy) > float(jump_threshold_m):
            score += 1.0
        if _grp_yaw_diff_deg(float(waypoints[i].yaw), float(waypoints[i - 1].yaw)) > float(yaw_flip_threshold_deg):
            score += 2.0
    return float(score)


def _compute_per_waypoint_speeds(
    waypoints: List[Waypoint],
    times: Optional[List[float]],
    default_dt: float = 0.1,
    window_seconds: float = 2.0,
    min_speed: float = 0.0,
    max_speed: float = 30.0,
    smooth_passes: int = 3,
) -> List[float]:
    """Compute a smoothed per-waypoint speed profile from trajectory + timing.

    For each waypoint the speed is estimated via forward *and* backward
    displacement windows of *window_seconds*, then averaged.  This cancels
    high-frequency tracking noise while preserving real acceleration and
    deceleration.  A lightweight rolling-average smoother is applied
    afterwards to remove any residual spikes.

    Returns a list of the same length as *waypoints* with speed in m/s.
    """
    n = len(waypoints)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    # Build time array
    if times and len(times) >= n:
        t = [float(ti) for ti in times[:n]]
    else:
        t = [float(default_dt) * i for i in range(n)]

    # --- Forward-window displacement speed ---
    fwd: List[float] = [0.0] * n
    j = 0
    for i in range(n):
        while j < n - 1 and (t[j] - t[i]) < window_seconds:
            j += 1
        dt_w = t[j] - t[i]
        if dt_w > 1e-6:
            dx = float(waypoints[j].x) - float(waypoints[i].x)
            dy = float(waypoints[j].y) - float(waypoints[i].y)
            fwd[i] = math.hypot(dx, dy) / dt_w

    # --- Backward-window displacement speed ---
    bwd: List[float] = [0.0] * n
    k = n - 1
    for i in range(n - 1, -1, -1):
        while k > 0 and (t[i] - t[k]) < window_seconds:
            k -= 1
        dt_w = t[i] - t[k]
        if dt_w > 1e-6:
            dx = float(waypoints[i].x) - float(waypoints[k].x)
            dy = float(waypoints[i].y) - float(waypoints[k].y)
            bwd[i] = math.hypot(dx, dy) / dt_w

    # Average forward and backward estimates
    raw: List[float] = []
    for i in range(n):
        if fwd[i] > 0 and bwd[i] > 0:
            raw.append(0.5 * (fwd[i] + bwd[i]))
        elif fwd[i] > 0:
            raw.append(fwd[i])
        elif bwd[i] > 0:
            raw.append(bwd[i])
        else:
            raw.append(0.0)

    # Clamp to [min_speed, max_speed]
    raw = [max(min_speed, min(max_speed, s)) for s in raw]

    # --- Rolling average smoother (kernel size 5) ---
    speeds = list(raw)
    k_half = 2  # kernel radius → window of 5
    for _ in range(smooth_passes):
        smoothed = list(speeds)
        for i in range(n):
            lo = max(0, i - k_half)
            hi = min(n, i + k_half + 1)
            smoothed[i] = sum(speeds[lo:hi]) / (hi - lo)
        speeds = smoothed

    # Final clamp
    speeds = [max(min_speed, min(max_speed, s)) for s in speeds]
    return speeds


def _stabilize_initial_route_yaw_for_export(
    waypoints: List[Waypoint],
    max_lookahead: int = 50,
    min_displacement_m: float = 1.0,
    max_prefix_frames: int = 50,
    stationary_prefix_tol_m: float = 0.8,
    apply_if_diff_deg: float = 35.0,
) -> List[Waypoint]:
    """Adjust yaws during stationary/slow-start period to avoid U-turns.

    This fixes cases where:
    1. Initial route waypoints have yaw pointing opposite to movement direction
    2. Tracking jitter causes yaw to flip 180° during stationary periods
    
    The function computes the actual movement direction from the first significant
    displacement, then corrects all yaws in the stationary prefix that differ
    by more than apply_if_diff_deg from that direction.
    """
    if len(waypoints) < 2:
        return waypoints

    out = [Waypoint(x=float(w.x), y=float(w.y), z=float(w.z), yaw=float(w.yaw)) for w in waypoints]
    w0 = out[0]

    # Find the first waypoint with significant displacement to determine movement direction
    end_idx = min(max_lookahead, len(out) - 1)
    ref_idx = -1
    ref_dx = 0.0
    ref_dy = 0.0
    for j in range(1, end_idx + 1):
        dx = float(out[j].x) - float(w0.x)
        dy = float(out[j].y) - float(w0.y)
        if math.hypot(dx, dy) >= min_displacement_m:
            ref_idx = j
            ref_dx = dx
            ref_dy = dy
            break

    if ref_idx < 0:
        return out

    spawn_heading = _normalize_yaw_deg(math.degrees(math.atan2(ref_dy, ref_dx)))

    # Fix all waypoints in the stationary prefix (not just first few)
    # A waypoint is "stationary" if it hasn't moved far from the spawn point
    prefix_end = min(max_prefix_frames, len(out))
    for i in range(prefix_end):
        di = math.hypot(float(out[i].x) - float(w0.x), float(out[i].y) - float(w0.y))
        if di > stationary_prefix_tol_m:
            # Once we've moved significantly, stop fixing yaws
            break
        yaw_diff = _grp_yaw_diff_deg(float(out[i].yaw), spawn_heading)
        if yaw_diff >= apply_if_diff_deg:
            out[i] = Waypoint(
                x=float(out[i].x),
                y=float(out[i].y),
                z=float(out[i].z),
                yaw=float(spawn_heading),
            )

    return out

def _spread_parked_vehicles(
    actor_tracks: List[Dict[str, object]],
    min_gap_m: float = 3.0,
) -> List[Dict[str, object]]:
    """
    Spread out parked/static vehicles that are too close together (bumper-to-bumper).
    
    For vehicles that are stationary (parked) and share similar headings (same lane),
    this function adds spacing along their heading axis to avoid collisions.
    
    Args:
        actor_tracks: List of actor track dictionaries
        min_gap_m: Minimum gap (in meters) between parked vehicles
    
    Returns:
        Modified actor_tracks with adjusted positions
    """
    if not actor_tracks:
        return actor_tracks
    
    # Identify parked/static vehicles with their positions
    parked_info = []
    for idx, track in enumerate(actor_tracks):
        is_parked = bool(track.get("parked_vehicle", False))
        role = str(track.get("role", "")).lower()
        model = str(track.get("model", "")).lower()
        frames = track.get("frames", [])
        
        # Check if it's a vehicle (not walker)
        is_vehicle = model.startswith("vehicle.") or role in ("npc", "static", "parked")
        is_walker = role in ("walker", "cyclist", "pedestrian")
        
        if is_walker:
            continue
        
        # Check if vehicle is stationary based on movement
        if not is_parked and len(frames) >= 2:
            first_f = frames[0]
            last_f = frames[-1]
            fx = float(first_f.get("rx", first_f.get("x", 0)))
            fy = float(first_f.get("ry", first_f.get("y", 0)))
            lx = float(last_f.get("rx", last_f.get("x", 0)))
            ly = float(last_f.get("ry", last_f.get("y", 0)))
            total_movement = math.sqrt((lx - fx) ** 2 + (ly - fy) ** 2)
            if total_movement < 1.5:  # Less than 1.5m total = essentially stationary
                is_parked = True
        
        if is_parked and is_vehicle and frames:
            f = frames[0]
            x = float(f.get("rx", f.get("x", 0)))
            y = float(f.get("ry", f.get("y", 0)))
            yaw = float(f.get("ryaw", f.get("yaw", 0)))
            parked_info.append({
                "idx": idx,
                "x": x,
                "y": y,
                "yaw": yaw,
            })
    
    if len(parked_info) < 2:
        return actor_tracks
    
    # Group by similar heading (within 30 degrees = same lane direction)
    heading_groups: Dict[int, List[dict]] = {}
    for pi in parked_info:
        # Normalize yaw to 0-180 range (treat opposite directions as same lane)
        norm_yaw = pi["yaw"] % 180
        group_key = int(norm_yaw / 30)  # Group by 30-degree buckets
        if group_key not in heading_groups:
            heading_groups[group_key] = []
        heading_groups[group_key].append(pi)
    
    # Process each heading group
    spread_count = 0
    for group_key, group in heading_groups.items():
        if len(group) < 2:
            continue
        
        # Calculate average heading for projection
        avg_yaw = sum(p["yaw"] for p in group) / len(group)
        cos_yaw = math.cos(math.radians(avg_yaw))
        sin_yaw = math.sin(math.radians(avg_yaw))
        
        # Project positions onto heading axis
        for p in group:
            p["proj"] = p["x"] * cos_yaw + p["y"] * sin_yaw
        
        # Sort by projection (position along heading direction)
        group.sort(key=lambda p: p["proj"])
        
        # Spread vehicles that are too close
        for i in range(1, len(group)):
            prev = group[i - 1]
            curr = group[i]
            
            gap = curr["proj"] - prev["proj"]
            
            if gap < min_gap_m:
                shift_needed = min_gap_m - gap + 0.5  # Extra buffer
                
                # Shift current vehicle forward along heading
                new_x = curr["x"] + shift_needed * cos_yaw
                new_y = curr["y"] + shift_needed * sin_yaw
                
                # Update all frames in this track
                orig_idx = curr["idx"]
                frames = actor_tracks[orig_idx].get("frames", [])
                for f in frames:
                    if "rx" in f:
                        f["rx"] = float(f["rx"]) + shift_needed * cos_yaw
                    if "ry" in f:
                        f["ry"] = float(f["ry"]) + shift_needed * sin_yaw
                    if "x" in f:
                        f["x"] = float(f["x"]) + shift_needed * cos_yaw
                    if "y" in f:
                        f["y"] = float(f["y"]) + shift_needed * sin_yaw
                
                # Update curr for next iteration
                curr["proj"] = prev["proj"] + min_gap_m
                curr["x"] = new_x
                curr["y"] = new_y
                spread_count += 1
                
                vid = actor_tracks[orig_idx].get("vid", orig_idx)
                print(f"[CARLA_EXPORT] Spread parked vehicle vid={vid} by {shift_needed:.2f}m to avoid bumper-to-bumper")
    
    if spread_count > 0:
        print(f"[CARLA_EXPORT] Spread {spread_count} parked vehicles to maintain {min_gap_m}m minimum gap")
    
    return actor_tracks


def export_carla_routes(
    out_dir: Path,
    town: str,
    route_id: str,
    ego_tracks: List[Dict[str, object]],
    actor_tracks: List[Dict[str, object]],
    align_cfg: Dict[str, object],
    ego_path_source: str = "auto",
    actor_control_mode: str = "policy",
    walker_control_mode: str = "policy",
    encode_timing: bool = True,
    snap_to_road: bool = False,
    static_spawn_only: bool = False,
    default_dt: float = 0.1,
) -> Dict[str, object]:
    """
    Export ego and actor trajectories to CARLA-compatible XML route files.
    
    Args:
        out_dir: Output directory for CARLA routes
        town: CARLA town name
        route_id: Route ID for XML files
        ego_tracks: List of ego track dictionaries from payload
        actor_tracks: List of actor track dictionaries from payload
        align_cfg: Alignment configuration (for V2XPNP -> CARLA transform)
        ego_path_source:
            Source for ego route geometry:
              - 'raw': use raw tracked poses (rx/ry/ryaw)
              - 'map': use map-snapped poses (mx/my/myaw)
              - 'corr': use lane-correspondence poses (cx/cy/cyaw), fallback to map if unavailable
              - 'auto': prefer map when valid, otherwise raw
        actor_control_mode: 'policy' for AI planners or 'replay' for exact replay
        walker_control_mode: 'policy' for AI walkers or 'replay' for exact replay
        encode_timing: Whether to include timing in waypoints
        snap_to_road: Whether to snap actors to road in CARLA
        static_spawn_only: For parked vehicles, only output spawn point
        default_dt: Default time step
    
    Returns:
        Report dictionary with export statistics
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    actors_dir = out_dir / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)
    
    # Spread out parked vehicles that are too close together (before coordinate transform)
    actor_tracks = _spread_parked_vehicles(actor_tracks, min_gap_m=3.0)
    
    # Coordinate transform parameters (V2XPNP -> CARLA)
    scale = float(align_cfg.get("scale", 1.0))
    theta_deg = float(align_cfg.get("theta_deg", 0.0))
    tx = float(align_cfg.get("tx", 0.0))
    ty = float(align_cfg.get("ty", 0.0))
    flip_y = bool(align_cfg.get("flip_y", False))
    inv_scale = 1.0 / scale if abs(scale) > 1e-12 else 1.0
    # Optional inverse compensation for lane-correspondence ICP refine.
    # runtime_projection may apply an extra rigid transform in V2 frame before
    # computing corr poses. Export must undo it before V2->CARLA conversion.
    _icp_raw = align_cfg.get("lane_corr_icp_refine")
    if not isinstance(_icp_raw, dict):
        _icp_raw = align_cfg.get("icp_refine")
    icp_applied = bool(_icp_raw.get("applied", False)) if isinstance(_icp_raw, dict) else False
    icp_theta_deg = float(_safe_float(_icp_raw.get("delta_theta_deg"), 0.0)) if isinstance(_icp_raw, dict) else 0.0
    icp_tx = float(_safe_float(_icp_raw.get("delta_tx"), 0.0)) if isinstance(_icp_raw, dict) else 0.0
    icp_ty = float(_safe_float(_icp_raw.get("delta_ty"), 0.0)) if isinstance(_icp_raw, dict) else 0.0
    icp_enabled = bool(
        icp_applied and (
            abs(icp_theta_deg) > 1e-6 or
            abs(icp_tx) > 1e-6 or
            abs(icp_ty) > 1e-6
        )
    )

    if icp_enabled:
        print(
            f"[EXPORT][ALIGN] Compensating corr poses for ICP refine: "
            f"dtheta={icp_theta_deg:.4f}deg dtx={icp_tx:.3f} dty={icp_ty:.3f}"
        )

    def v2x_to_carla(x: float, y: float) -> Tuple[float, float]:
        cx, cy = invert_se2((x, y), theta_deg, tx, ty, flip_y=flip_y)
        return cx * inv_scale, cy * inv_scale

    def _undo_icp_in_v2(x: float, y: float) -> Tuple[float, float]:
        if not icp_enabled:
            return float(x), float(y)
        # Inverse of p_corr = R(theta) @ p_base + t.
        px = float(x) - float(icp_tx)
        py = float(y) - float(icp_ty)
        th = math.radians(float(icp_theta_deg))
        c = math.cos(th)
        s = math.sin(th)
        bx = c * px + s * py
        by = -s * px + c * py
        return float(bx), float(by)

    def _apply_icp_in_v2(x: float, y: float) -> Tuple[float, float]:
        if not icp_enabled:
            return float(x), float(y)
        th = math.radians(float(icp_theta_deg))
        c = math.cos(th)
        s = math.sin(th)
        cx = c * float(x) - s * float(y) + float(icp_tx)
        cy = s * float(x) + c * float(y) + float(icp_ty)
        return float(cx), float(cy)

    def v2x_corr_to_carla(x: float, y: float) -> Tuple[float, float]:
        bx, by = _undo_icp_in_v2(float(x), float(y))
        return v2x_to_carla(bx, by)

    def carla_to_v2x(x: float, y: float) -> Tuple[float, float]:
        sx, sy = float(x) * scale, float(y) * scale
        return apply_se2((sx, sy), theta_deg, tx, ty, flip_y=flip_y)
    
    def yaw_v2x_to_carla(yaw_v2x: float) -> float:
        adjusted = float(yaw_v2x) - float(theta_deg)
        if flip_y:
            adjusted = -adjusted
        return _normalize_yaw_deg(adjusted)

    def yaw_carla_to_v2x(yaw_carla: float) -> float:
        adjusted = float(yaw_carla)
        if flip_y:
            adjusted = -adjusted
        return _normalize_yaw_deg(adjusted + float(theta_deg))

    def _candidate_max_step_xy(wps: List[Waypoint]) -> float:
        if len(wps) < 2:
            return 0.0
        max_step = 0.0
        for i in range(len(wps) - 1):
            a = wps[i]
            b = wps[i + 1]
            step = math.hypot(float(b.x) - float(a.x), float(b.y) - float(a.y))
            if step > max_step:
                max_step = step
        return float(max_step)
    
    report: Dict[str, object] = {
        "enabled": True,
        "output_dir": str(out_dir),
        "town": town,
        "route_id": route_id,
        "ego_path_source_requested": str(ego_path_source),
        "ego_path_sources": {},
        "actor_control_mode": actor_control_mode,
        "walker_control_mode": walker_control_mode,
        "ego_count": 0,
        "npc_count": 0,
        "walker_count": 0,
        "static_count": 0,
        "total_actors": 0,
        "ego_files": [],
        "actor_files": [],
        "actor_path_sources": {},
    }
    
    manifest: Dict[str, List[Dict[str, object]]] = {}

    requested_ego_source = str(ego_path_source or "auto").strip().lower()
    if requested_ego_source not in ("auto", "raw", "map", "corr"):
        requested_ego_source = "auto"
    
    # --- Export Ego Routes ---
    ego_entries: List[Dict[str, object]] = []
    for ego_idx, ego_track in enumerate(ego_tracks):
        frames = ego_track.get("frames", [])
        if not frames:
            continue
        
        raw_wps: List[Waypoint] = []
        raw_times: List[float] = []
        map_wps: List[Waypoint] = []
        map_times: List[float] = []
        corr_wps: List[Waypoint] = []
        corr_times: List[float] = []
        corr_valid = True

        # Detect whether the dataset actually has dedicated mx/my fields
        # (as opposed to silently falling back to rx/x).  When mx/my don't
        # exist, "map" source is identical to "raw" and should not be
        # preferred over the lane-snapped correspondence source (cx/cy).
        _has_explicit_mx = any("mx" in f for f in frames[:min(10, len(frames))])

        for f in frames:
            rx = float(f.get("rx", f.get("x", 0)))
            ry = float(f.get("ry", f.get("y", 0)))
            rz = float(f.get("rz", f.get("z", 0)))
            ryaw = float(f.get("ryaw", f.get("yaw", 0)))
            mx = _safe_float(f.get("mx"), rx)
            my = _safe_float(f.get("my"), ry)
            t = float(f.get("t", 0))

            raw_cx, raw_cy = v2x_to_carla(rx, ry)
            raw_cyaw = yaw_v2x_to_carla(ryaw)
            raw_wps.append(Waypoint(x=raw_cx, y=raw_cy, z=rz, yaw=raw_cyaw))
            raw_times.append(t)

            map_cx, map_cy = v2x_to_carla(mx, my)
            # Keep raw yaw for ego to avoid lane-direction ambiguity (map/corr
            # lanes can be oriented opposite to trajectory direction).
            map_cyaw = yaw_v2x_to_carla(ryaw)
            map_wps.append(Waypoint(x=map_cx, y=map_cy, z=rz, yaw=map_cyaw))
            map_times.append(t)

            if corr_valid:
                sx = _safe_float(f.get("cx"), float("nan"))
                sy = _safe_float(f.get("cy"), float("nan"))
                syaw = _safe_float(f.get("cyaw"), float("nan"))
                if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(syaw)):
                    corr_valid = False
                    corr_wps = []
                    corr_times = []
                else:
                    corr_cx, corr_cy = v2x_corr_to_carla(float(sx), float(sy))
                    # Keep raw yaw for ego even when using correspondence XY.
                    corr_cyaw = yaw_v2x_to_carla(ryaw)
                    corr_wps.append(Waypoint(x=corr_cx, y=corr_cy, z=rz, yaw=corr_cyaw))
                    corr_times.append(t)

        if not raw_wps:
            continue

        selected_source = "raw"
        carla_wps = raw_wps
        carla_times = raw_times
        corr_ready = corr_valid and len(corr_wps) == len(raw_wps)
        map_valid = _candidate_max_step_xy(map_wps) <= 2.5
        corr_valid_continuity = corr_ready and (_candidate_max_step_xy(corr_wps) <= 2.5)

        if requested_ego_source == "raw":
            selected_source = "raw"
            carla_wps, carla_times = raw_wps, raw_times
        elif requested_ego_source == "map":
            if map_valid:
                selected_source = "map"
                carla_wps, carla_times = map_wps, map_times
            else:
                selected_source = "raw_fallback"
                carla_wps, carla_times = raw_wps, raw_times
                print(
                    f"[EXPORT][EGO] ego_{ego_idx}: map requested but continuity check failed; "
                    "falling back to raw poses."
                )
        elif requested_ego_source == "corr":
            if corr_valid_continuity:
                selected_source = "corr"
                carla_wps, carla_times = corr_wps, corr_times
            else:
                if map_valid:
                    selected_source = "map_fallback"
                    carla_wps, carla_times = map_wps, map_times
                else:
                    selected_source = "raw_fallback"
                    carla_wps, carla_times = raw_wps, raw_times
                print(
                    f"[EXPORT][EGO] ego_{ego_idx}: corr requested but unavailable/unstable; "
                    "falling back to a safer source."
                )
        else:
            # Auto mode: prefer the source that best matches the
            # dashboard-visible positions.
            #  1. corr (cx/cy) — lane-snapped correspondence positions
            #     (what the dashboard displays when available)
            #  2. map  (mx/my) — only if genuinely distinct from raw
            #  3. raw  (x/y)   — original tracking positions
            if corr_valid_continuity:
                selected_source = "corr"
                carla_wps, carla_times = corr_wps, corr_times
            elif _has_explicit_mx and map_valid:
                selected_source = "map"
                carla_wps, carla_times = map_wps, map_times
            else:
                selected_source = "raw"
                carla_wps, carla_times = raw_wps, raw_times

        # ── SAFETY CHECK: detect if we're about to write raw (unsnapped)
        # coordinates when lane-snapped correspondence (cx/cy) was available.
        # This almost certainly means the source-selection logic silently fell
        # back to raw tracking, which produces vehicles offset from lane centers.
        if corr_ready and len(corr_wps) == len(raw_wps) and len(carla_wps) == len(raw_wps):
            _is_using_raw = all(
                abs(carla_wps[i].x - raw_wps[i].x) < 1e-6
                and abs(carla_wps[i].y - raw_wps[i].y) < 1e-6
                for i in range(min(20, len(carla_wps)))
            )
            _corr_differs = any(
                abs(corr_wps[i].x - raw_wps[i].x) > 0.01
                or abs(corr_wps[i].y - raw_wps[i].y) > 0.01
                for i in range(min(20, len(corr_wps)))
            )
            if _is_using_raw and _corr_differs:
                raise RuntimeError(
                    f"[EXPORT][EGO] ego_{ego_idx}: ABORTING — selected source "
                    f"'{selected_source}' produced coordinates identical to raw "
                    f"tracking (x/y), but lane-snapped correspondence (cx/cy) was "
                    f"available and differs by >0.01m.  This means vehicles will "
                    f"be offset from lane centers in CARLA.  "
                    f"Fix: use ego_path_source='corr' or ensure 'auto' mode "
                    f"prefers correspondence when available."
                )

        print(
            f"[EXPORT][EGO] ego_{ego_idx}: source={selected_source} "
            f"(requested={requested_ego_source})"
        )
        if isinstance(report.get("ego_path_sources"), dict):
            report["ego_path_sources"][f"ego_{ego_idx}"] = str(selected_source)
        
        ego_xml_name = f"{town.lower()}_custom_ego_vehicle_{ego_idx}.xml"
        ego_xml_path = out_dir / ego_xml_name
        
        _write_carla_route_xml(
            path=ego_xml_path,
            route_id=route_id,
            role="ego",
            town=town,
            waypoints=carla_wps,
            times=carla_times if encode_timing else None,
            snap_to_road=False,  # Ego should follow exact route
            control_mode="",  # Ego controlled by agent
        )
        
        ego_entry = {
            "file": ego_xml_name,
            "route_id": route_id,
            "town": town,
            "name": f"ego_{ego_idx}",
            "kind": "ego",
            "model": str(ego_track.get("model", "vehicle.lincoln.mkz_2020")),
        }
        ego_entries.append(ego_entry)
        report["ego_files"].append(ego_xml_name)
        report["ego_count"] = int(report["ego_count"]) + 1
    
    manifest["ego"] = ego_entries
    
    # --- Export Actor Routes (grouped by kind) ---
    actors_by_kind: Dict[str, List[Dict[str, object]]] = {}
    
    for actor_track in actor_tracks:
        actor_id = actor_track.get("id", "unknown")
        vid = actor_track.get("vid", 0)
        role = str(actor_track.get("role", "npc")).lower()
        model = str(actor_track.get("model", ""))
        is_parked = bool(actor_track.get("parked_vehicle", False))
        frames = actor_track.get("frames", [])
        
        if not frames:
            continue
        
        # Determine kind and control mode
        if role == "cyclist":
            kind = "cyclist"
            control = walker_control_mode
        elif role == "walker":
            kind = "walker"
            control = walker_control_mode
        elif is_parked:
            kind = "static"
            control = "replay"  # Static actors always use replay (stationary)
            if not model:
                model = "vehicle.tesla.model3"  # fallback so route_parser doesn't spawn a traffic cone
        else:
            kind = "npc"
            control = actor_control_mode
        control = str(control).strip().lower()
        
        # Build candidate source tracks in V2X coordinates.
        # RAW keeps highest geometric fidelity, while correspondence-projected
        # poses can suppress A<->B lane jitter for policy NPCs.
        raw_wps: List[Waypoint] = []
        raw_times: List[float] = []
        corr_wps: List[Waypoint] = []
        corr_times: List[float] = []
        corr_valid = True

        for f in frames:
            rz = float(f.get("rz", f.get("z", 0)))
            t = float(f.get("t", 0))

            # RAW source (always available)
            rx = float(f.get("rx", f.get("x", 0)))
            ry = float(f.get("ry", f.get("y", 0)))
            ryaw = float(f.get("ryaw", f.get("yaw", 0)))
            raw_cx, raw_cy = v2x_to_carla(rx, ry)
            raw_cyaw = yaw_v2x_to_carla(ryaw)
            raw_wps.append(Waypoint(x=raw_cx, y=raw_cy, z=rz, yaw=raw_cyaw))
            raw_times.append(t)

            # Correspondence source (optional)
            if corr_valid:
                if ("cx" not in f) or ("cy" not in f) or ("cyaw" not in f):
                    corr_valid = False
                else:
                    sx = _safe_float(f.get("cx"), float("nan"))
                    sy = _safe_float(f.get("cy"), float("nan"))
                    syaw = _safe_float(f.get("cyaw"), float("nan"))
                    if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(syaw)):
                        corr_valid = False
                    else:
                        corr_cx, corr_cy = v2x_corr_to_carla(float(sx), float(sy))
                        corr_cyaw = yaw_v2x_to_carla(float(syaw))
                        corr_wps.append(Waypoint(x=corr_cx, y=corr_cy, z=rz, yaw=corr_cyaw))
                        corr_times.append(t)

        if not raw_wps:
            continue

        # Choose source trajectory:
        # - default RAW
        # - for replay vehicles (npc/static), prefer correspondence so exported
        #   XML geometry matches dashboard-visible snapped trajectories
        # - for policy NPCs, compare RAW vs correspondence and switch only when
        #   correspondence is materially more stable (less jumpy / fewer 180 flips).
        selected_source = "raw"
        selected_wps: List[Waypoint] = raw_wps
        selected_times: List[float] = raw_times
        selected_orig_len = len(raw_wps)

        def _postprocess_export_track(src_wps: List[Waypoint], src_times: List[float]) -> Tuple[List[Waypoint], List[float], float]:
            wps_local = [Waypoint(x=float(w.x), y=float(w.y), z=float(w.z), yaw=float(w.yaw)) for w in src_wps]
            times_local = [float(ti) for ti in src_times]
            # Skip sanitization in replay mode — we need frame-perfect fidelity
            # to preserve exact positions, timing, and speed at every timestep.
            if control != "replay" and kind in ("npc", "walker") and len(wps_local) >= 3:
                # Use a lower speed threshold for walkers (no walker goes >10 m/s)
                san_speed = 40.0 if kind == "npc" else 10.0
                wps_local, times_local = _sanitize_trajectory(
                    wps_local,
                    times_local,
                    default_dt,
                    max_plausible_speed_mps=san_speed,
                )
            # Run stabilization AFTER sanitization so one noisy frame does not
            # bias spawn heading and force a wrong-way initialization.
            # Skip for replay mode — preserve exact dataset yaw values.
            if control != "replay" and kind in ("npc", "walker"):
                wps_local = _stabilize_initial_route_yaw_for_export(wps_local)
            score_local = _trajectory_instability_score(wps_local)
            return wps_local, times_local, float(score_local)

        raw_proc_wps, raw_proc_times, raw_score = _postprocess_export_track(raw_wps, raw_times)
        selected_wps, selected_times = raw_proc_wps, raw_proc_times

        if kind in ("npc", "static") and control == "replay":
            # Replay mode for vehicles: prefer correspondence poses whenever
            # available so XML routes match dashboard rendering (which uses cx/cy).
            if corr_valid and len(corr_wps) == len(raw_wps):
                corr_proc_wps, corr_proc_times, _ = _postprocess_export_track(corr_wps, corr_times)
                selected_source = "corr"
                selected_wps, selected_times = corr_proc_wps, corr_proc_times
                selected_orig_len = len(corr_wps)
            else:
                selected_source = "raw"
                # (raw_proc_wps already assigned above)
        elif kind == "npc" and control == "policy" and corr_valid and len(corr_wps) == len(raw_wps):
            corr_proc_wps, corr_proc_times, corr_score = _postprocess_export_track(corr_wps, corr_times)
            if corr_score + 2.0 < raw_score:
                selected_source = "corr"
                selected_wps, selected_times = corr_proc_wps, corr_proc_times
                selected_orig_len = len(corr_wps)
                print(
                    f"[EXPORT] {actor_id}: using correspondence poses "
                    f"(instability {raw_score:.1f} -> {corr_score:.1f})"
                )

        carla_wps = selected_wps
        carla_times = selected_times

        # ── SAFETY CHECK: detect if we're about to write raw (unsnapped)
        # coordinates for replay vehicle actors when lane-snapped corr exists.
        if (corr_valid and len(corr_wps) == len(raw_wps)
                and len(carla_wps) == len(raw_proc_wps)
                and kind in ("npc", "static")):
            _is_using_raw_actor = all(
                abs(carla_wps[i].x - raw_proc_wps[i].x) < 1e-6
                and abs(carla_wps[i].y - raw_proc_wps[i].y) < 1e-6
                for i in range(min(20, len(carla_wps)))
            )
            _corr_differs_actor = any(
                abs(corr_wps[i].x - raw_wps[i].x) > 0.01
                or abs(corr_wps[i].y - raw_wps[i].y) > 0.01
                for i in range(min(20, len(corr_wps)))
            )
            if _is_using_raw_actor and _corr_differs_actor:
                raise RuntimeError(
                    f"[EXPORT][VEHICLE] {actor_id}: ABORTING — selected source "
                    f"'{selected_source}' produced coordinates identical to raw "
                    f"tracking (x/y), but lane-snapped correspondence (cx/cy) was "
                    f"available and differs.  Vehicles will be offset from lane "
                    f"centers in CARLA.  Fix the source-selection logic."
                )

        if kind in ("npc", "walker") and len(carla_wps) != selected_orig_len:
            print(
                f"[SANITIZE] {actor_id} ({selected_source}): "
                f"{selected_orig_len} -> {len(carla_wps)} waypoints after cleaning"
            )

        # For static actors with spawn_only mode, keep only the first waypoint
        if kind == "static" and static_spawn_only and carla_wps:
            carla_wps = [carla_wps[0]]
            carla_times = [carla_times[0]] if carla_times else []

        # Attach export/runtime metadata to payload track so the viewer can
        # render replay actors using the exact route exported to CARLA.
        actor_track["carla_kind"] = str(kind)
        actor_track["carla_control_mode"] = str(control)
        actor_track["carla_path_source"] = str(selected_source)
        if isinstance(report.get("actor_path_sources"), dict):
            report["actor_path_sources"][str(actor_id)] = {
                "kind": str(kind),
                "control_mode": str(control),
                "source": str(selected_source),
                "waypoint_count": int(len(carla_wps)),
            }
        if kind != "walker" and control == "replay":
            exec_frames: List[Dict[str, float]] = []
            for idx_wp, wp in enumerate(carla_wps):
                vx, vy = carla_to_v2x(float(wp.x), float(wp.y))
                if selected_source == "corr":
                    vx, vy = _apply_icp_in_v2(vx, vy)
                vyaw = yaw_carla_to_v2x(float(wp.yaw))
                if idx_wp < len(carla_times):
                    vt = float(carla_times[idx_wp])
                else:
                    vt = float(idx_wp) * float(default_dt)
                exec_frames.append(
                    {
                        "t": float(round(vt, 6)),
                        "x": float(vx),
                        "y": float(vy),
                        "z": float(wp.z),
                        "yaw": float(vyaw),
                    }
                )
            actor_track["carla_exec_frames"] = exec_frames
        else:
            actor_track.pop("carla_exec_frames", None)
        
        # Compute per-waypoint speeds and global fallback target speed
        per_wp_speeds: Optional[List[float]] = None
        target_speed: Optional[float] = None
        if control == "policy" and len(carla_wps) >= 2:
            per_wp_speeds = _compute_per_waypoint_speeds(carla_wps, carla_times, default_dt)
            target_speed = _compute_target_speed(carla_wps, carla_times, default_dt)
        
        # Create kind subdirectory
        kind_dir = actors_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        actor_type = role.replace(" ", "_").replace("-", "_").title()
        if not actor_type or actor_type.lower() == "npc":
            actor_type = "Vehicle"
        xml_name = f"{town.lower()}_custom_{actor_type}_{vid}_{kind}.xml"
        xml_path = kind_dir / xml_name
        
        _write_carla_route_xml(
            path=xml_path,
            route_id=route_id,
            role=kind,
            town=town,
            waypoints=carla_wps,
            times=carla_times if encode_timing else None,
            snap_to_road=snap_to_road and kind == "npc",
            control_mode=control,
            target_speed_mps=target_speed,
            model=model,
            speeds=per_wp_speeds,
        )
        
        # Build manifest entry
        actor_entry: Dict[str, object] = {
            "file": f"actors/{kind}/{xml_name}",
            "route_id": route_id,
            "town": town,
            "name": str(actor_id),
            "kind": kind,
            "model": model,
            "control_mode": control,
        }
        if target_speed is not None and target_speed > 0:
            actor_entry["target_speed"] = round(target_speed, 2)
            # route_parser.py reads "speed" (not "target_speed"), so emit both.
            actor_entry["speed"] = round(target_speed, 2)
        
        # Add to manifest by kind
        if kind not in actors_by_kind:
            actors_by_kind[kind] = []
        actors_by_kind[kind].append(actor_entry)
        report["actor_files"].append(f"actors/{kind}/{xml_name}")
        
        # Update counts
        if kind == "npc":
            report["npc_count"] = int(report["npc_count"]) + 1
        elif kind in ("walker", "cyclist"):
            report["walker_count"] = int(report["walker_count"]) + 1
        elif kind == "static":
            report["static_count"] = int(report["static_count"]) + 1
    
    # Merge actor manifest entries
    for kind, entries in actors_by_kind.items():
        manifest[kind] = entries
    
    report["total_actors"] = (
        int(report["npc_count"]) + 
        int(report["walker_count"]) + 
        int(report["static_count"])
    )
    
    # Write manifest
    manifest_path = out_dir / "actors_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    # Write control mode configuration for run_custom_eval.py
    control_cfg = {
        "ego_path_source": requested_ego_source,
        "actor_control_mode": actor_control_mode,
        "walker_control_mode": walker_control_mode,
        "encode_timing": encode_timing,
        "snap_to_road": snap_to_road,
        "static_spawn_only": static_spawn_only,
        "town": town,
        "route_id": route_id,
        "manifest_path": "actors_manifest.json",
        "ego_count": report["ego_count"],
        "npc_count": report["npc_count"],
        "walker_count": report["walker_count"],
        "static_count": report["static_count"],
    }
    control_cfg_path = out_dir / "carla_control_config.json"
    control_cfg_path.write_text(json.dumps(control_cfg, indent=2), encoding="utf-8")
    
    print(f"[CARLA-EXPORT] Exported CARLA routes to: {out_dir}")
    print(
        f"[CARLA-EXPORT]   Ego: {report['ego_count']}, "
        f"NPC (control={actor_control_mode}): {report['npc_count']}, "
        f"Walker (control={walker_control_mode}): {report['walker_count']}, "
        f"Static: {report['static_count']}"
    )
    
    return report


def main() -> None:
    global _active_carla_manager
    args = parse_args()
    
    # --- Auto-launch CARLA server if requested ---
    carla_manager: Optional[CarlaProcessManager] = None
    if bool(args.start_carla):
        # Determine CARLA root directory
        carla_root_str = args.carla_root
        if not carla_root_str:
            carla_root_str = os.environ.get("CARLA_ROOT")
        if not carla_root_str:
            # Default: look for carla912 relative to workspace root
            workspace_root = Path(__file__).resolve().parent.parent.parent
            carla_root_str = str(workspace_root / "carla912")
        carla_root = Path(carla_root_str).expanduser().resolve()
        
        if not carla_root.exists():
            raise SystemExit(f"CARLA root not found: {carla_root}")
        
        carla_manager = CarlaProcessManager(
            carla_root=carla_root,
            host=str(args.carla_host),
            port=int(args.carla_port),
            extra_args=list(args.carla_arg),
            port_tries=int(args.carla_port_tries),
            port_step=int(args.carla_port_step),
        )
        
        # Install signal handlers for clean shutdown
        _install_carla_signal_handlers()
        _active_carla_manager = carla_manager
        
        # Start CARLA and update port if it changed
        actual_port = carla_manager.start()
        if actual_port != int(args.carla_port):
            args.carla_port = actual_port
            print(f"[INFO] CARLA port updated to: {actual_port}")
    
    try:
        # Check if batch mode or single mode
        if args.scenario_dirs:
            _run_batch_processing(args, carla_manager)
        elif args.scenario_dir:
            _run_main_logic(args)
        else:
            raise SystemExit("Either --scenario-dir or --scenario-dirs must be provided.")
    finally:
        # Ensure CARLA is stopped on exit
        if carla_manager is not None:
            carla_manager.stop()
            _active_carla_manager = None


def _is_scenario_directory(path: Path) -> bool:
    """
    Check if a directory is a valid scenario directory.
    
    A scenario directory typically contains YAML subdirectories with vehicle/actor data.
    """
    if not path.is_dir():
        return False
    
    # Check for common scenario indicators
    yaml_indicators = ['yaml_to_carla_log', 'yaml_data', 'vehicle_data', 'actor_data']
    
    # Check if any subdirectory matches scenario patterns
    for subdir in path.iterdir():
        if subdir.is_dir():
            # Check for numbered subdirectories (common in scenarios)
            if subdir.name.isdigit():
                return True
            # Check for yaml-related subdirectories
            if any(ind in subdir.name.lower() for ind in yaml_indicators):
                return True
            # Check if subdir contains .yaml files
            try:
                yaml_files = list(subdir.glob('*.yaml')) + list(subdir.glob('*.yml'))
                if yaml_files:
                    return True
            except PermissionError:
                pass
    
    # Check for direct yaml files
    try:
        yaml_files = list(path.glob('*.yaml')) + list(path.glob('*.yml'))
        if yaml_files:
            return True
    except PermissionError:
        pass
    
    return False


def _expand_scenario_directories(paths: List[Path]) -> List[Path]:
    """
    Expand a list of paths to find all valid scenario directories.
    
    If a path is a parent directory containing multiple scenarios, expand it.
    If a path is a scenario directory itself, keep it.
    
    Returns:
        List of scenario directories (deduplicated)
    """
    expanded = []
    seen = set()
    
    for path in paths:
        if not path.exists():
            continue
            
        if _is_scenario_directory(path):
            # This is a scenario directory
            if path not in seen:
                expanded.append(path)
                seen.add(path)
        else:
            # Check if this is a parent directory containing scenarios
            try:
                for subdir in sorted(path.iterdir()):
                    if subdir.is_dir() and _is_scenario_directory(subdir):
                        if subdir not in seen:
                            expanded.append(subdir)
                            seen.add(subdir)
            except PermissionError:
                pass
    
    return expanded


def _run_batch_processing(args: argparse.Namespace, carla_manager: Optional[CarlaProcessManager] = None) -> None:
    """
    Process multiple scenario directories in batch mode.
    
    Each scenario directory is processed, simulated, aligned, and has results/videos generated.
    Results are named after each scenario folder for easy identification.
    
    Supports:
    - Multiple explicit scenario directories
    - Parent directories containing multiple scenarios (auto-expanded)
    - Handles duplicate scenario names by adding unique suffixes
    """
    from datetime import datetime
    
    input_paths = [Path(d).expanduser().resolve() for d in args.scenario_dirs]
    
    # Validate and report missing directories
    for path in input_paths:
        if not path.exists():
            print(f"[WARN] Path not found, skipping: {path}")
    
    existing_paths = [p for p in input_paths if p.exists()]
    if not existing_paths:
        raise SystemExit("No valid paths found.")
    
    # Expand to find all scenario directories
    scenario_dirs = _expand_scenario_directories(existing_paths)
    if not scenario_dirs:
        print("[INFO] No scenario directories found directly. Checking if inputs are parent directories...")
        # Try treating each path as a parent and find scenarios inside
        for path in existing_paths:
            if path.is_dir():
                print(f"[INFO] Scanning: {path}")
                for subdir in sorted(path.iterdir()):
                    if subdir.is_dir():
                        print(f"  - {subdir.name}: {'scenario' if _is_scenario_directory(subdir) else 'not a scenario'}")
        raise SystemExit("No valid scenario directories found. Check that directories contain YAML data.")
    
    print(f"[INFO] Found {len(scenario_dirs)} scenario directories to process")
    
    # Determine batch results root
    batch_results_root = None
    if args.batch_results_root:
        batch_results_root = Path(args.batch_results_root).expanduser().resolve()
    else:
        # Default: create a batch_results folder in the parent of the first scenario
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_results_root = scenario_dirs[0].parent / f"batch_results_{timestamp}"
    batch_results_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING MODE")
    print(f"{'='*60}")
    print(f"Total scenarios: {len(scenario_dirs)}")
    print(f"Results root: {batch_results_root}")
    print(f"{'='*60}\n")
    
    # Track scenario names to handle duplicates
    seen_names: Dict[str, int] = {}
    
    batch_report = {
        "start_time": datetime.now().isoformat(),
        "scenarios": [],
        "success_count": 0,
        "failure_count": 0,
        "total_count": len(scenario_dirs),
    }
    
    for idx, scenario_dir in enumerate(scenario_dirs, 1):
        base_scenario_name = scenario_dir.name
        
        # Handle duplicate scenario names by adding a suffix
        if base_scenario_name in seen_names:
            seen_names[base_scenario_name] += 1
            scenario_name = f"{base_scenario_name}_{seen_names[base_scenario_name]}"
        else:
            seen_names[base_scenario_name] = 0
            scenario_name = base_scenario_name
        
        print(f"\n{'='*60}")
        print(f"PROCESSING SCENARIO {idx}/{len(scenario_dirs)}: {scenario_name}")
        print(f"  Source: {scenario_dir}")
        print(f"{'='*60}\n")
        
        scenario_result = {
            "name": scenario_name,
            "base_name": base_scenario_name,
            "path": str(scenario_dir),
            "success": False,
            "error": None,
            "output_dir": None,
            "video_paths": [],
        }
        
        try:
            # Create scenario-specific output directory
            scenario_out_dir = batch_results_root / scenario_name
            scenario_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a copy of args for this scenario
            scenario_args = argparse.Namespace(**vars(args))
            scenario_args.scenario_dir = str(scenario_dir)
            scenario_args.out_dir = str(scenario_out_dir)
            
            # Run the main logic for this scenario
            _run_main_logic(scenario_args)
            
            scenario_result["success"] = True
            scenario_result["output_dir"] = str(scenario_out_dir)
            batch_report["success_count"] += 1
            
            # Generate videos if requested
            if args.generate_videos:
                video_paths = _generate_scenario_videos(
                    scenario_dir,
                    scenario_name,
                    fps=float(args.video_fps),
                    resize_factor=int(getattr(args, 'video_resize_factor', 2)),
                )
                scenario_result["video_paths"] = video_paths
            
            print(f"\n[OK] Scenario {scenario_name} completed successfully.")
            
        except Exception as exc:
            scenario_result["error"] = str(exc)
            batch_report["failure_count"] += 1
            print(f"\n[ERROR] Scenario {scenario_name} failed: {exc}")
            import traceback
            traceback.print_exc()
        
        batch_report["scenarios"].append(scenario_result)
    
    batch_report["end_time"] = datetime.now().isoformat()
    
    # Write batch report
    report_path = batch_results_root / "batch_report.json"
    report_path.write_text(json.dumps(batch_report, indent=2), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {batch_report['success_count']}/{batch_report['total_count']}")
    print(f"Failures: {batch_report['failure_count']}/{batch_report['total_count']}")
    print(f"Results: {batch_results_root}")
    print(f"Report: {report_path}")
    print(f"{'='*60}\n")


def _ego_indices(scenario_dir: Path) -> List[int]:
    """Return sorted positive-integer ego indices found as subdirectories of *scenario_dir*.

    The dataset convention uses 1-indexed subdirectories (``1/``, ``2/``, …) for each ego
    vehicle's camera images.  Negative folders (e.g. ``-1/``) are infrastructure and are skipped.
    """
    indices: List[int] = []
    try:
        for child in scenario_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name.strip()
            if not name.isdigit():
                continue
            idx = int(name)
            if idx > 0:
                indices.append(idx)
    except OSError:
        pass
    return sorted(set(indices))


def _pick_run_image_dir(image_root: Path) -> Optional[Path]:
    """Pick the most-recently-modified image run subfolder under *image_root*.

    ``run_custom_eval.py`` stores captured images in a timestamped subdirectory
    under ``<results>/<scenario>/image/<timestamp>/``.  This helper picks the newest one.
    """
    if not image_root.exists():
        return None
    dirs = [p for p in image_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _generate_scenario_videos(
    scenario_dir: Path,
    scenario_name: str,
    fps: float = 10.0,
    resize_factor: int = 2,
    fullvideos_dir: Optional[Path] = None,
    results_root: Optional[Path] = None,
    video_conda_env: Optional[str] = None,
) -> List[str]:
    """Generate per-ego side-by-side, CARLA-only and real-only videos.

    Mirrors the standard side-by-side video generation stage used in log-replay flows:
    all videos are written into a central *fullvideos_dir* named
    ``{scenario_name}_{variant}_{ego_id}.mp4``.

    It invokes ``visualization/gen_video.py`` exactly as the batch pipeline does.

    Args:
        scenario_dir: Dataset scenario directory (contains ``1/``, ``2/``, … ego camera folders).
        scenario_name: Scenario identifier used for video filenames.
        fps: Frames per second for the output videos.
        resize_factor: Down-scale factor passed to ``gen_video.py``.
        fullvideos_dir: Output directory for all videos.
            Defaults to ``results/results_driving_custom/fullvideos``.
        results_root: Root of the evaluation results tree.
            Defaults to ``results/results_driving_custom``.
        video_conda_env: Optional conda environment name to run gen_video.py under.

    Returns:
        List of paths to the generated mp4 files.
    """
    repo_root = Path(__file__).resolve().parents[2]
    gen_video_script = repo_root / "visualization" / "gen_video.py"
    if not gen_video_script.exists():
        print(f"[WARN] gen_video.py not found at {gen_video_script}; skipping video generation.")
        return []

    if results_root is None:
        results_root = repo_root / "results" / "results_driving_custom"
    if fullvideos_dir is None:
        fullvideos_dir = results_root / "fullvideos"
    fullvideos_dir.mkdir(parents=True, exist_ok=True)

    # Discover the image run directory created by run_custom_eval.
    image_root = results_root / scenario_name / "image"
    run_image_dir = _pick_run_image_dir(image_root)
    if run_image_dir is None:
        print(f"[WARN] No image run directory found under {image_root}; skipping video generation.")
        return []
    print(f"[INFO] image run dir: {run_image_dir}")

    # Discover ego indices from the dataset scenario directory.
    ego_ids = _ego_indices(scenario_dir)
    if not ego_ids:
        print(f"[WARN] No positive ego-index folders found in {scenario_dir}; skipping video step.")
        return []

    # Build the python command prefix.
    python_bin = sys.executable

    def _build_cmd(script_args: List[str]) -> List[str]:
        if video_conda_env:
            cmd = ["conda", "run", "-n", video_conda_env, "python", "-u", str(gen_video_script)]
        else:
            cmd = [python_bin, "-u", str(gen_video_script)]
        cmd.extend(script_args)
        return cmd

    generated: List[str] = []

    for ego_id in ego_ids:
        real_cam_dir = scenario_dir / str(ego_id)
        # Logreplay images use 0-indexed ego IDs; dataset folders use 1-indexed.
        # Try both naming conventions: rgb_front_N (logreplay_agent) and logreplay_rgb_N (tcp_agent).
        carla_img_dir: Optional[Path] = None
        logreplay_base = run_image_dir / "logreplayimages"
        for candidate_name in [
            f"rgb_front_{ego_id - 1}",
            f"logreplay_rgb_{ego_id - 1}",
        ]:
            candidate = logreplay_base / candidate_name
            if candidate.exists():
                carla_img_dir = candidate
                break

        out_side = fullvideos_dir / f"{scenario_name}_sidebyside_{ego_id}.mp4"
        out_carla = fullvideos_dir / f"{scenario_name}_carla_{ego_id}.mp4"
        out_real = fullvideos_dir / f"{scenario_name}_real_{ego_id}.mp4"

        real_exists = real_cam_dir.exists()
        carla_exists = carla_img_dir is not None and carla_img_dir.exists()

        if not real_exists:
            print(f"[WARN] Missing real cam folder for ego {ego_id}: {real_cam_dir}")
        if not carla_exists:
            print(f"[WARN] Missing logreplay folder for ego {ego_id}")

        jobs: List[tuple] = []

        # Side-by-side: real (cam1) + CARLA
        if real_exists and carla_exists and carla_img_dir is not None:
            jobs.append((
                "sidebyside",
                out_side,
                _build_cmd([
                    str(real_cam_dir),
                    "--only-suffix", "cam1",
                    "--side-by-side-dir", str(carla_img_dir),
                    "--fps", str(fps),
                    "--resize-factor", str(resize_factor),
                    "--output", str(out_side),
                ]),
            ))

        # CARLA-only
        if carla_exists and carla_img_dir is not None:
            jobs.append((
                "carla",
                out_carla,
                _build_cmd([
                    str(carla_img_dir),
                    "--fps", str(fps),
                    "--resize-factor", str(resize_factor),
                    "--output", str(out_carla),
                ]),
            ))

        # Real-only (cam1)
        if real_exists:
            jobs.append((
                "real",
                out_real,
                _build_cmd([
                    str(real_cam_dir),
                    "--only-suffix", "cam1",
                    "--fps", str(fps),
                    "--resize-factor", str(resize_factor),
                    "--output", str(out_real),
                ]),
            ))

        if not jobs:
            print(f"[WARN] No video inputs available for ego {ego_id}; skipping.")
            continue

        for tag, out_mp4, cmd in jobs:
            print(f"[INFO] Generating {tag} video for ego {ego_id}: {out_mp4.name}")
            try:
                subprocess.run(cmd, check=True)
                if out_mp4.exists() and out_mp4.stat().st_size > 0:
                    generated.append(str(out_mp4))
                    print(f"[OK] {tag} video: {out_mp4}")
                else:
                    print(f"[WARN] {tag} video was not created or is empty.")
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] {tag} video generation failed (exit {exc.returncode})")
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] {tag} video generation error: {exc}")

    if generated:
        print(f"[INFO] All videos written to: {fullvideos_dir}")

    return generated


def _run_main_logic(args: argparse.Namespace) -> None:
    """Main processing logic, separated for clean CARLA lifecycle management."""
    # skip_map_snap_compute only affects _apply_lane_correspondence_to_payload; CARLA layer still loads
    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    if not scenario_dir.exists():
        raise SystemExit(f"Scenario directory not found: {scenario_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (scenario_dir / "yaml_map_export")
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_dirs = pick_yaml_dirs(scenario_dir, args.subdir)
    if not yaml_dirs:
        raise SystemExit(f"No YAML dirs found under: {scenario_dir}")

    print(f"[INFO] Selected YAML dirs ({len(yaml_dirs)}):")
    for yd in yaml_dirs:
        print(f"  - {yd}")

    vehicles, vehicle_times, ego_trajs, ego_times, obj_info, actor_source_subdir, actor_orig_vid, merge_stats = _merge_subdir_trajectories(
        yaml_dirs=yaml_dirs,
        dt=float(args.dt),
        id_merge_distance_m=float(args.id_merge_distance_m),
    )
    actor_alias_vids: Dict[int, List[int]] = {int(vid): [int(vid)] for vid in vehicles.keys()}
    if bool(args.cross_id_dedup):
        (
            vehicles,
            vehicle_times,
            obj_info,
            actor_source_subdir,
            actor_orig_vid,
            actor_alias_vids,
            cross_id_stats,
        ) = _deduplicate_cross_id_tracks(
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            obj_info=obj_info,
            actor_source_subdir=actor_source_subdir,
            actor_orig_vid=actor_orig_vid,
            dt=float(args.dt),
            max_median_dist_m=float(args.cross_id_dedup_max_median_dist_m),
            max_p90_dist_m=float(args.cross_id_dedup_max_p90_dist_m),
            max_median_yaw_diff_deg=float(args.cross_id_dedup_max_median_yaw_diff_deg),
            min_common_points=int(args.cross_id_dedup_min_common_points),
            min_overlap_ratio_each=float(args.cross_id_dedup_min_overlap_each),
            min_overlap_ratio_any=float(args.cross_id_dedup_min_overlap_any),
        )
    else:
        cross_id_stats = {
            "cross_id_dedup_enabled": False,
            "cross_id_pair_checks": 0,
            "cross_id_candidate_pairs": 0,
            "cross_id_clusters": 0,
            "cross_id_removed": 0,
            "cross_id_removed_ids": [],
        }
    merge_stats.update(cross_id_stats)
    merge_stats["output_tracks"] = int(len(vehicles))

    timing_optimization: Dict[str, object] = {
        "timing_policy": {
            "spawn": "first_observed_frame",
            "despawn": "last_observed_frame",
            "global_early_spawn_optimization": False,
            "global_late_despawn_optimization": False,
        },
        "early_spawn": {
            "enabled": bool(args.maximize_safe_early_spawn),
            "applied": False,
            "reason": "disabled_by_flag" if not bool(args.maximize_safe_early_spawn) else "not_run",
            "adjusted_actor_ids": [],
            "adjusted_spawn_times": {},
        },
        "late_despawn": {
            "enabled": bool(args.maximize_safe_late_despawn),
            "applied": False,
            "reason": "disabled_by_flag" if not bool(args.maximize_safe_late_despawn) else "not_run",
            "adjusted_actor_ids": [],
            "hold_until_time": 0.0,
        },
    }

    early_safety_margin = max(0.30, float(args.early_spawn_safety_margin))
    late_safety_margin = max(0.30, float(args.late_despawn_safety_margin))

    actor_meta_for_timing = _build_actor_meta_for_timing_optimization(vehicles, obj_info)
    timing_vehicles = vehicles
    timing_vehicle_times = vehicle_times
    timing_actor_meta = actor_meta_for_timing
    timing_blocker_labels: Dict[int, str] = {}
    if actor_meta_for_timing:
        timing_vehicles, timing_vehicle_times, timing_actor_meta, timing_blocker_labels = (
            _augment_timing_inputs_with_ego_blockers(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                actor_meta=actor_meta_for_timing,
                ego_trajs=ego_trajs,
                ego_times=ego_times,
                dt=float(args.dt),
            )
        )
        if timing_blocker_labels:
            print(
                f"[INFO] Timing optimization blockers: "
                f"{len(timing_blocker_labels)} ego trajectories included as non-adjustable safety blockers."
            )

    if bool(args.maximize_safe_early_spawn):
        if actor_meta_for_timing:
            selected_spawn_times, early_report = _maximize_safe_early_spawn_actors(
                vehicles=timing_vehicles,
                vehicle_times=timing_vehicle_times,
                actor_meta=timing_actor_meta,
                dt=float(args.dt),
                safety_margin=float(early_safety_margin),
            )
            selected_spawn_times = {
                int(vid): float(t)
                for vid, t in selected_spawn_times.items()
                if int(vid) in vehicles
            }
            vehicles, vehicle_times, adjusted_ids, applied_spawn_times = _apply_early_spawn_time_overrides(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                early_spawn_times=selected_spawn_times,
                dt=float(args.dt),
            )
            early_report = dict(early_report)
            early_report["enabled"] = True
            early_report["applied"] = True
            early_report["reason"] = "applied"
            early_report["safety_margin_m"] = float(early_safety_margin)
            early_report["adjusted_actor_ids"] = [int(v) for v in adjusted_ids]
            early_report["adjusted_spawn_times"] = {
                str(int(vid)): float(t) for vid, t in sorted(applied_spawn_times.items())
            }
            if timing_blocker_labels:
                early_report["timing_blocker_count"] = int(len(timing_blocker_labels))
                early_report["timing_blockers"] = {
                    str(int(bid)): str(lbl)
                    for bid, lbl in sorted(timing_blocker_labels.items())
                }
                blocked_examples = early_report.get("blocked_examples")
                if isinstance(blocked_examples, list):
                    for item in blocked_examples:
                        if not isinstance(item, dict):
                            continue
                        bid = _safe_int(item.get("blocked_by"), 0)
                        if int(bid) in timing_blocker_labels:
                            item["blocked_by_label"] = str(timing_blocker_labels[int(bid)])
            timing_optimization["early_spawn"] = early_report
            timing_optimization["timing_policy"]["global_early_spawn_optimization"] = True
        else:
            timing_optimization["early_spawn"] = {
                "enabled": True,
                "applied": False,
                "reason": "no_actor_metadata",
                "adjusted_actor_ids": [],
                "adjusted_spawn_times": {},
            }

    if bool(args.maximize_safe_late_despawn):
        if actor_meta_for_timing:
            hold_until_time = 0.0
            for times in timing_vehicle_times.values():
                if times:
                    hold_until_time = max(float(hold_until_time), float(times[-1]))
            late_report: Dict[str, object] = {
                "enabled": True,
                "applied": False,
                "hold_until_time": float(hold_until_time),
                "adjusted_actor_ids": [],
            }
            if hold_until_time > 0.0:
                selected_ids, select_report = _maximize_safe_late_despawn_actors(
                    vehicles=timing_vehicles,
                    vehicle_times=timing_vehicle_times,
                    actor_meta=timing_actor_meta,
                    dt=float(args.dt),
                    safety_margin=float(late_safety_margin),
                    hold_until_time=float(hold_until_time),
                )
                selected_ids = {int(v) for v in selected_ids if int(v) in vehicles}
                vehicles, vehicle_times, adjusted_ids = _apply_late_despawn_time_overrides(
                    vehicles=vehicles,
                    vehicle_times=vehicle_times,
                    selected_late_hold_ids=selected_ids,
                    dt=float(args.dt),
                    hold_until_time=float(hold_until_time),
                )
                late_report.update(dict(select_report))
                late_report["applied"] = True
                late_report["reason"] = "applied"
                late_report["safety_margin_m"] = float(late_safety_margin)
                late_report["adjusted_actor_ids"] = [int(v) for v in adjusted_ids]
                if timing_blocker_labels:
                    late_report["timing_blocker_count"] = int(len(timing_blocker_labels))
                    late_report["timing_blockers"] = {
                        str(int(bid)): str(lbl)
                        for bid, lbl in sorted(timing_blocker_labels.items())
                    }
                    blocked_examples = late_report.get("blocked_examples")
                    if isinstance(blocked_examples, list):
                        for item in blocked_examples:
                            if not isinstance(item, dict):
                                continue
                            bid = _safe_int(item.get("blocked_by"), 0)
                            if int(bid) in timing_blocker_labels:
                                item["blocked_by_label"] = str(timing_blocker_labels[int(bid)])
                timing_optimization["timing_policy"]["global_late_despawn_optimization"] = True
            else:
                late_report["reason"] = "non_positive_horizon"
            timing_optimization["late_despawn"] = late_report
        else:
            timing_optimization["late_despawn"] = {
                "enabled": True,
                "applied": False,
                "reason": "no_actor_metadata",
                "adjusted_actor_ids": [],
                "hold_until_time": 0.0,
            }

    print(
        f"[INFO] Parsed trajectories: egos={len(ego_trajs)} actors={len(vehicles)} "
        f"(timed={sum(1 for v in vehicle_times.values() if v)}) "
        f"id_collisions={merge_stats['ids_with_collisions']} "
        f"merged={merge_stats['merged_duplicates']} "
        f"split={merge_stats['split_tracks_created']} "
        f"cross_id_removed={merge_stats.get('cross_id_removed', 0)} "
        f"cross_id_clusters={merge_stats.get('cross_id_clusters', 0)} "
        f"early_adjusted={len(timing_optimization.get('early_spawn', {}).get('adjusted_actor_ids', []))} "
        f"late_adjusted={len(timing_optimization.get('late_despawn', {}).get('adjusted_actor_ids', []))}"
    )

    map_paths: List[Path] = []
    if args.map_pkl:
        map_paths = [Path(p).expanduser().resolve() for p in args.map_pkl]
    else:
        map_paths = [
            Path("/data2/marco/CoLMDriver/v2xpnp/map/v2v_corridors_vector_map.pkl"),
            Path("/data2/marco/CoLMDriver/v2xpnp/map/v2x_intersection_vector_map.pkl"),
        ]
    for p in map_paths:
        if not p.exists():
            raise SystemExit(f"Map pickle not found: {p}")

    map_data_list = [_load_vector_map(p) for p in map_paths]
    chosen_map, selection_scores = _select_best_map(
        maps=map_data_list,
        ego_trajs=ego_trajs,
        vehicles=vehicles,
        sample_count=int(args.map_selection_sample_count),
        bbox_margin=float(args.map_selection_bbox_margin),
    )
    print(f"[INFO] Selected map: {chosen_map.name} ({chosen_map.source_path})")
    for score in selection_scores:
        print(
            "  - {name}: score={score:.3f} median={median_nearest_m:.2f}m "
            "p90={p90_nearest_m:.2f}m outside={outside_bbox_ratio:.3f}".format(**score)
        )

    carla_map_layer: Optional[Dict[str, object]] = None
    if bool(args.carla_map_layer):
        carla_cache_path = Path(args.carla_map_cache).expanduser().resolve()
        carla_align_path = Path(args.carla_map_offset_json).expanduser().resolve() if args.carla_map_offset_json else None
        if not carla_cache_path.exists():
            print(f"[WARN] CARLA map cache not found; skipping CARLA layer: {carla_cache_path}")
        else:
            try:
                raw_lines, cache_bounds, cache_map_name = _load_carla_map_cache_lines(carla_cache_path)
                align_cfg = _load_carla_alignment_cfg(carla_align_path)
                transformed_lines, transformed_bbox = _transform_carla_lines(raw_lines, align_cfg)
                if transformed_lines:
                    carla_map_layer = {
                        "name": str(cache_map_name or "carla_westwood_map"),
                        "source_path": str(carla_cache_path),
                        "alignment_path": str(carla_align_path) if carla_align_path is not None else "",
                        "raw_bounds": {
                            "min_x": float(cache_bounds[0]),
                            "max_x": float(cache_bounds[1]),
                            "min_y": float(cache_bounds[2]),
                            "max_y": float(cache_bounds[3]),
                        }
                        if cache_bounds is not None
                        else None,
                        "bbox": {
                            "min_x": float(transformed_bbox[0]),
                            "max_x": float(transformed_bbox[1]),
                            "min_y": float(transformed_bbox[2]),
                            "max_y": float(transformed_bbox[3]),
                        },
                        "lines": transformed_lines,
                        "transform": {
                            "scale": float(align_cfg.get("scale", 1.0)),
                            "theta_deg": float(align_cfg.get("theta_deg", 0.0)),
                            "tx": float(align_cfg.get("tx", 0.0)),
                            "ty": float(align_cfg.get("ty", 0.0)),
                            "flip_y": bool(align_cfg.get("flip_y", False)),
                            "source_path": str(align_cfg.get("source_path", "")),
                        },
                    }
                    print(
                        f"[INFO] Loaded CARLA map layer: lines={len(transformed_lines)} "
                        f"source={carla_cache_path} align={carla_align_path if carla_align_path else '-'}"
                    )
                else:
                    print(f"[WARN] CARLA map cache had no valid polylines after transform: {carla_cache_path}")
            except Exception as exc:
                print(f"[WARN] Failed to build CARLA map layer from cache: {exc}")

    # --- Load or capture CARLA top-down image underlay ---
    if carla_map_layer is not None:
        img_cache_path = Path(args.carla_map_image_cache).expanduser().resolve()
        img_meta_path = img_cache_path.with_suffix(".json")
        # raw_bounds from the CARLA map cache
        raw_b = carla_map_layer.get("raw_bounds")
        raw_bounds_tuple: Optional[Tuple[float, float, float, float]] = None
        if isinstance(raw_b, dict):
            try:
                raw_bounds_tuple = (
                    float(raw_b["min_x"]),
                    float(raw_b["max_x"]),
                    float(raw_b["min_y"]),
                    float(raw_b["max_y"]),
                )
            except Exception:
                pass
        result = _load_or_capture_carla_topdown(
            image_cache_path=img_cache_path,
            meta_cache_path=img_meta_path,
            carla_host=str(args.carla_host),
            carla_port=int(args.carla_port),
            raw_bounds=raw_bounds_tuple,
            capture_enabled=bool(args.capture_carla_image),
            carla_map_name=str(args.carla_map_name),
            method=str(args.carla_topdown_method),
            ortho_altitude=float(args.carla_topdown_altitude),
            ortho_fov_deg=float(args.carla_topdown_fov_deg),
            ortho_image_px=int(args.carla_topdown_image_px),
            ortho_waypoint_spacing_m=float(args.carla_topdown_waypoint_spacing_m),
            ortho_px_per_meter=float(args.carla_topdown_px_per_meter),
        )
        if result is not None:
            jpeg_bytes, img_raw_bounds = result
            img_b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
            # Transform image bounds into V2XPNP coordinate space
            img_transform = carla_map_layer.get("transform", {})
            img_bounds_v2 = _transform_image_bounds_to_v2xpnp(img_raw_bounds, img_transform)
            carla_map_layer["image_b64"] = img_b64_str
            carla_map_layer["image_bounds"] = img_bounds_v2
            print(
                f"[INFO] CARLA top-down image attached: "
                f"{len(jpeg_bytes)} bytes, b64={len(img_b64_str)} chars, "
                f"bounds={img_bounds_v2}"
            )
        else:
            print("[INFO] No CARLA top-down image available (skipping underlay).")

    # --- GRP-aware trajectory alignment (before map-matching / export) ---
    grp_align_report: Dict[str, object] = {"enabled": False}
    if bool(args.grp_align):
        grp_align_cfg_path = (
            Path(args.carla_map_offset_json).expanduser().resolve()
            if args.carla_map_offset_json
            else None
        )
        grp_align_cfg = _load_carla_alignment_cfg(grp_align_cfg_path)
        vehicles, vehicle_times, ego_trajs, ego_times, grp_align_report = _grp_align_trajectories(
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            ego_trajs=ego_trajs,
            ego_times=ego_times,
            obj_info=obj_info,
            parked_vehicle_cfg={
                "net_disp_max_m": float(args.parked_net_disp_max_m),
                "radius_p90_max_m": float(args.parked_radius_p90_max_m),
                "radius_max_m": float(args.parked_radius_max_m),
                "p95_step_max_m": float(args.parked_p95_step_max_m),
                "max_from_start_m": float(args.parked_max_from_start_m),
                "large_step_threshold_m": float(args.parked_large_step_threshold_m),
                "large_step_ratio_max": float(args.parked_large_step_max_ratio),
                "robust_cluster_enabled": float(1.0 if bool(args.parked_robust_cluster) else 0.0),
                "robust_cluster_eps_m": float(args.parked_robust_cluster_eps_m),
                "robust_min_inlier_ratio": float(args.parked_robust_min_inlier_ratio),
                "robust_max_outlier_run": float(args.parked_robust_max_outlier_run),
                "robust_min_points": float(args.parked_robust_min_points),
            },
            align_cfg=grp_align_cfg,
            carla_host=str(args.carla_host),
            carla_port=int(args.carla_port),
            carla_map_name=str(args.carla_map_name),
            sampling_resolution=float(args.grp_sampling_resolution),
            snap_radius=float(args.grp_snap_radius),
            snap_k=int(args.grp_snap_k),
            heading_thresh=float(args.grp_heading_thresh),
            lane_change_penalty=float(args.grp_lane_change_penalty),
            default_dt=float(args.dt),
            actor_max_median_displacement_m=float(args.grp_actor_max_median_displacement_m),
            actor_max_p90_displacement_m=float(args.grp_actor_max_p90_displacement_m),
            actor_max_displacement_m=float(args.grp_actor_max_displacement_m),
            ego_max_median_displacement_m=float(args.grp_ego_max_median_displacement_m),
            ego_max_p90_displacement_m=float(args.grp_ego_max_p90_displacement_m),
            ego_max_displacement_m=float(args.grp_ego_max_displacement_m),
            enabled=True,
        )

    matcher = LaneMatcher(chosen_map)
    lane_change_cfg: Dict[str, object] = {
        "enabled": bool(args.lane_change_filter),
        "lane_top_k": int(args.lane_snap_top_k),
        "confirm_window": int(args.lane_change_confirm_window),
        "confirm_votes": int(args.lane_change_confirm_votes),
        "cooldown_frames": int(args.lane_change_cooldown_frames),
        "endpoint_guard_frames": int(args.lane_change_endpoint_guard_frames),
        "endpoint_extra_votes": int(args.lane_change_endpoint_extra_votes),
        "min_improvement_m": float(args.lane_change_min_improvement_m),
        "keep_lane_max_dist": float(args.lane_change_keep_lane_max_dist),
        "short_run_max": int(args.lane_change_short_run_max),
        "endpoint_short_run": int(args.lane_change_endpoint_short_run),
    }
    vehicle_lane_policy_cfg: Dict[str, object] = {
        "forbidden_lane_types": str(args.vehicle_forbidden_lane_types),
        "parked_only_lane_types": str(args.vehicle_parked_only_lane_types),
    }
    parked_vehicle_cfg: Dict[str, float] = {
        "net_disp_max_m": float(args.parked_net_disp_max_m),
        "radius_p90_max_m": float(args.parked_radius_p90_max_m),
        "radius_max_m": float(args.parked_radius_max_m),
        "p95_step_max_m": float(args.parked_p95_step_max_m),
        "max_from_start_m": float(args.parked_max_from_start_m),
        "large_step_threshold_m": float(args.parked_large_step_threshold_m),
        "large_step_ratio_max": float(args.parked_large_step_max_ratio),
        "robust_cluster_enabled": float(1.0 if bool(args.parked_robust_cluster) else 0.0),
        "robust_cluster_eps_m": float(args.parked_robust_cluster_eps_m),
        "robust_min_inlier_ratio": float(args.parked_robust_min_inlier_ratio),
        "robust_max_outlier_run": float(args.parked_robust_max_outlier_run),
        "robust_min_points": float(args.parked_robust_min_points),
    }

    # --- Walker sidewalk compression and spawn stabilization ---
    walker_processing_report: Dict[str, object] = {"enabled": False}
    if bool(args.walker_sidewalk_compression):
        # Get CARLA map lines (road polylines in V2XPNP coordinate space)
        carla_lines_for_walker: List[List[List[float]]] = []
        if carla_map_layer is not None:
            carla_lines_for_walker = carla_map_layer.get("lines", [])
        
        if not carla_lines_for_walker:
            print("[WARN] Walker sidewalk compression: no CARLA map lines available, skipping.")
            walker_processing_report = {
                "enabled": False,
                "reason": "no_carla_map_lines",
            }
        else:
            print(f"[INFO] Processing walker sidewalk compression using {len(carla_lines_for_walker)} CARLA road polylines...")
            walker_processor = WalkerSidewalkProcessor(
                carla_map_lines=carla_lines_for_walker,
                lane_spacing_m=args.walker_lane_spacing_m,  # None = auto-calibrate
                sidewalk_start_factor=float(args.walker_sidewalk_start_factor),
                sidewalk_outer_factor=float(args.walker_sidewalk_outer_factor),
                compression_target_band_m=float(args.walker_compression_target_band_m),
                compression_power=float(args.walker_compression_power),
                min_spawn_separation_m=float(args.walker_min_spawn_separation_m),
                walker_radius_m=float(args.walker_radius_m),
                crossing_road_ratio_thresh=float(args.walker_crossing_road_ratio_thresh),
                crossing_lateral_thresh_m=float(args.walker_crossing_lateral_thresh_m),
                road_presence_min_frames=int(args.walker_road_presence_min_frames),
                max_lateral_offset_m=float(args.walker_max_lateral_offset_m),
                dt=float(args.dt),
            )
            vehicles, walker_processing_report = walker_processor.process_walkers(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                obj_info=obj_info,
            )
            # Print summary
            if walker_processing_report.get("walker_count", 0) > 0:
                stationary_removed = walker_processing_report.get("stationary_removed_count", 0)
                cls_summary = walker_processing_report.get("classification_summary", {})
                comp_summary = walker_processing_report.get("compression_summary", {})
                stab = walker_processing_report.get("stabilization", {})
                print(
                    f"[INFO] Walker processing: {walker_processing_report.get('walker_count', 0)} walkers | "
                    f"stationary/jitter removed: {stationary_removed} | "
                    f"classified: sidewalk={cls_summary.get('sidewalk_consistent', 0)}, "
                    f"crossing={cls_summary.get('crossing', 0)}, "
                    f"jaywalking={cls_summary.get('jaywalking', 0)}, "
                    f"road_walking={cls_summary.get('road_walking', 0)} | "
                    f"compressed: {comp_summary.get('compressed', 0)}, "
                    f"spawn-stabilized: {stab.get('adjusted_count', 0)}"
                )
                if float(comp_summary.get("max_lateral_offset", 0)) > 0:
                    print(
                        f"  [COMPRESS] avg_offset={comp_summary.get('avg_lateral_offset', 0):.2f}m, "
                        f"max_offset={comp_summary.get('max_lateral_offset', 0):.2f}m, "
                        f"lane_spacing={walker_processing_report.get('lane_spacing_m', 0):.1f}m"
                    )
            else:
                print("[INFO] Walker processing: no walkers found in trajectory data.")

    payload = _build_export_payload(
        scenario_dir=scenario_dir,
        selected_map=chosen_map,
        selection_details=selection_scores,
        ego_trajs=ego_trajs,
        ego_times=ego_times,
        vehicles=vehicles,
        vehicle_times=vehicle_times,
        obj_info=obj_info,
        actor_source_subdir=actor_source_subdir,
        actor_orig_vid=actor_orig_vid,
        actor_alias_vids=actor_alias_vids,
        merge_stats=merge_stats,
        timing_optimization=timing_optimization,
        matcher=matcher,
        snap_to_map=bool(args.snap_to_map),
        map_max_points_per_line=int(args.map_max_points_per_line),
        lane_change_cfg=lane_change_cfg,
        vehicle_lane_policy_cfg=vehicle_lane_policy_cfg,
        parked_vehicle_cfg=parked_vehicle_cfg,
        carla_map_layer=carla_map_layer,
        default_dt=float(args.dt),
    )
    if bool(args.lane_correspondence) and bool(payload.get("carla_map")):
        print("[INFO] Building lane correspondence (Hungarian assignment)...")
        driving_types = sorted(
            _parse_lane_type_set(
                args.lane_correspondence_driving_types,
                fallback=["1"],
            )
        )
        # Resolve cache directory — default is alongside this script
        _corr_cache_dir_raw = args.lane_correspondence_cache_dir
        if _corr_cache_dir_raw == "__script_dir__":
            _corr_cache_dir = Path(__file__).resolve().parent / "lane_corr_cache"
        elif _corr_cache_dir_raw == "__output_dir__":
            _corr_cache_dir = out_dir / "lane_corr_cache"
        elif _corr_cache_dir_raw:
            _corr_cache_dir = Path(_corr_cache_dir_raw).expanduser().resolve()
        else:
            _corr_cache_dir = None
        correspondence = _build_lane_correspondence(
            payload=payload,
            candidate_top_k=int(args.lane_correspondence_top_k),
            driving_lane_types=driving_types,
            cache_dir=_corr_cache_dir,
        )
        if bool(getattr(args, "skip_map_snap_compute", False)):
            print("[INFO] Skipping per-frame CARLA projection (--skip-map-snap-compute). 'Use map snapped poses' disabled.")
        else:
            print("[INFO] Starting per-frame CARLA projection from lane correspondence...")
            _corr_proj_t0 = time.monotonic()
            _apply_lane_correspondence_to_payload(payload, correspondence)
            print(f"[INFO] Per-frame CARLA projection done in {time.monotonic() - _corr_proj_t0:.2f}s")
        lc_meta = payload.get("metadata", {}).get("lane_correspondence", {})
        if bool(lc_meta.get("enabled", False)):
            print(
                "[INFO] Lane correspondence: "
                f"mapped_lanes={int(lc_meta.get('mapped_lane_count', 0))} "
                f"usable={int(lc_meta.get('usable_lane_count', 0))} "
                f"mapped_carla_lines={int(lc_meta.get('mapped_carla_line_count', 0))} "
                f"quality={lc_meta.get('quality_counts', {})} "
                f"splits={int(lc_meta.get('split_merge_count', 0))} "
                f"phantom_changes={int(lc_meta.get('total_phantom_lane_changes', 0))} "
                f"conn_preserved={int(lc_meta.get('connectivity_edges_preserved', 0))}/{int(lc_meta.get('connectivity_edges_total', 0))}"
            )
        else:
            print(f"[WARN] Lane correspondence disabled or failed: {lc_meta}")

    # --- CARLA Route XML Export (optional) ---
    carla_export_report: Dict[str, object] = {"enabled": False}
    if bool(args.export_carla_routes):
        _export_t0 = time.monotonic()
        # Retrieve alignment config for coordinate transform
        grp_align_cfg_path = (
            Path(args.carla_map_offset_json).expanduser().resolve()
            if args.carla_map_offset_json
            else None
        )
        grp_align_cfg = _load_carla_alignment_cfg(grp_align_cfg_path)
        
        # Determine output directory for CARLA routes
        carla_routes_out = (
            Path(args.carla_routes_dir).expanduser().resolve()
            if args.carla_routes_dir
            else out_dir / "carla_routes"
        )
        
        carla_export_report = export_carla_routes(
            ego_tracks=payload.get("ego_tracks", []),
            actor_tracks=payload.get("actor_tracks", []),
            align_cfg=grp_align_cfg,
            out_dir=carla_routes_out,
            town=str(args.carla_town),
            route_id=str(args.carla_route_id),
            ego_path_source=str(args.carla_ego_path_source),
            actor_control_mode=str(args.carla_actor_control_mode),
            walker_control_mode=str(args.carla_walker_control_mode),
            encode_timing=bool(args.carla_encode_timing),
            snap_to_road=bool(args.carla_snap_to_road),
            static_spawn_only=bool(args.carla_static_spawn_only),
            default_dt=float(args.dt),
        )
        print(f"[INFO] CARLA route export stage done in {time.monotonic() - _export_t0:.2f}s")

    if bool(carla_export_report.get("enabled", False)):
        # Keep viewer playback in sync with the exported replay trajectories.
        _refresh_payload_timeline_for_carla_exec(payload)
        payload.setdefault("metadata", {})["carla_route_export"] = {
            "enabled": True,
            "actor_control_mode": str(carla_export_report.get("actor_control_mode", "")),
            "walker_control_mode": str(carla_export_report.get("walker_control_mode", "")),
            "ego_path_sources": dict(carla_export_report.get("ego_path_sources", {}) or {}),
            "actor_path_sources": dict(carla_export_report.get("actor_path_sources", {}) or {}),
        }

    print("[INFO] Serializing replay data JSON...")
    _json_t0 = time.monotonic()
    data_json_path = out_dir / "yaml_map_replay_data.json"
    data_json_path.write_text(json.dumps(_sanitize_for_json(payload), indent=2), encoding="utf-8")
    print(f"[OK] Wrote replay data JSON: {data_json_path} ({time.monotonic() - _json_t0:.2f}s)")

    print("[INFO] Building interactive HTML viewer...")
    _html_t0 = time.monotonic()
    html_path = out_dir / "yaml_map_replay_viewer.html"
    html_path.write_text(_build_html(payload), encoding="utf-8")
    print(f"[OK] Wrote interactive HTML: {html_path} ({time.monotonic() - _html_t0:.2f}s)")

    summary_path = out_dir / "yaml_map_selection_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scenario_dir": str(scenario_dir),
                "selected_map": chosen_map.name,
                "selected_map_path": chosen_map.source_path,
                "map_selection_scores": selection_scores,
                "snap_to_map": bool(args.snap_to_map),
                "id_merge_stats": merge_stats,
                "timing_optimization": timing_optimization,
                "grp_alignment": {
                    k: v for k, v in grp_align_report.items() if k != "actor_details"
                },
                "walker_sidewalk_processing": {
                    "enabled": bool(walker_processing_report.get("enabled", False)),
                    "walker_count": int(walker_processing_report.get("walker_count", 0)),
                    "lane_spacing_m": float(walker_processing_report.get("lane_spacing_m", 0.0)),
                    "classification_summary": walker_processing_report.get("classification_summary", {}),
                    "compression_summary": walker_processing_report.get("compression_summary", {}),
                    "stabilization_summary": {
                        k: v for k, v in walker_processing_report.get("stabilization", {}).items()
                        if k != "details"
                    },
                },
                "vehicle_lane_policy": vehicle_lane_policy_cfg,
                "parked_vehicle_cfg": parked_vehicle_cfg,
                "lane_correspondence": payload.get("metadata", {}).get("lane_correspondence", {}),
                "carla_map_layer": {
                    "enabled": bool(carla_map_layer is not None),
                    "source_path": str(carla_map_layer.get("source_path")) if carla_map_layer else "",
                    "alignment_path": str(carla_map_layer.get("alignment_path")) if carla_map_layer else "",
                    "line_count": int(len(carla_map_layer.get("lines", []))) if carla_map_layer else 0,
                },
                "carla_route_export": {
                    "enabled": bool(carla_export_report.get("enabled", False)),
                    "output_dir": str(carla_export_report.get("output_dir", "")),
                    "town": str(carla_export_report.get("town", "")),
                    "route_id": str(carla_export_report.get("route_id", "")),
                    "ego_path_source_requested": str(carla_export_report.get("ego_path_source_requested", "")),
                    "ego_path_sources": dict(carla_export_report.get("ego_path_sources", {}) or {}),
                    "actor_path_sources": dict(carla_export_report.get("actor_path_sources", {}) or {}),
                    "actor_control_mode": str(carla_export_report.get("actor_control_mode", "")),
                    "walker_control_mode": str(carla_export_report.get("walker_control_mode", "")),
                    "ego_count": int(carla_export_report.get("ego_count", 0)),
                    "npc_count": int(carla_export_report.get("npc_count", 0)),
                    "walker_count": int(carla_export_report.get("walker_count", 0)),
                    "static_count": int(carla_export_report.get("static_count", 0)),
                    "total_actors": int(carla_export_report.get("total_actors", 0)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Wrote selection summary: {summary_path}")
    if bool(grp_align_report.get("enabled", False)):
        grp_report_path = out_dir / "grp_alignment_report.json"
        grp_report_path.write_text(json.dumps(grp_align_report, indent=2, default=str), encoding="utf-8")
        print(f"[OK] Wrote GRP alignment report: {grp_report_path}")
    if bool(walker_processing_report.get("enabled", False)) and walker_processing_report.get("walker_count", 0) > 0:
        walker_report_path = out_dir / "walker_sidewalk_report.json"
        walker_report_path.write_text(json.dumps(walker_processing_report, indent=2, default=str), encoding="utf-8")
        print(f"[OK] Wrote walker sidewalk report: {walker_report_path}")
    if args.timing_optimization_report:
        timing_report_path = Path(args.timing_optimization_report).expanduser().resolve()
        timing_report_path.parent.mkdir(parents=True, exist_ok=True)
        timing_report_path.write_text(json.dumps(timing_optimization, indent=2), encoding="utf-8")
        print(f"[OK] Wrote timing optimization report: {timing_report_path}")
    print("[DONE] yaml_to_map export complete.")
    
    # --- Automatically run CARLA scenario with exported routes ---
    if bool(args.run_custom_eval) and bool(carla_export_report.get("enabled", False)):
        carla_routes_dir = Path(carla_export_report.get("output_dir", ""))
        if carla_routes_dir.exists():
            # Derive scenario name from scenario_dir for results folder naming
            scenario_name = scenario_dir.name
            _run_carla_scenario(
                routes_dir=carla_routes_dir,
                port=int(args.eval_port) if args.eval_port else int(args.carla_port),
                planner=str(args.eval_planner) if args.eval_planner else None,
                overwrite=bool(args.eval_overwrite),
                actor_control_mode=str(args.carla_actor_control_mode),
                walker_control_mode=str(args.carla_walker_control_mode),
                capture_logreplay_images=bool(getattr(args, 'capture_logreplay_images', False)),
                capture_every_sensor_frame=bool(getattr(args, 'capture_every_sensor_frame', False)),
                npc_only_fake_ego=bool(getattr(args, "npc_only_fake_ego", False)),
                scenario_name=scenario_name,
            )
            
            # Generate videos if requested (single scenario mode)
            if bool(getattr(args, 'generate_videos', False)):
                print(f"\n[INFO] Generating videos for scenario: {scenario_name}")
                video_paths = _generate_scenario_videos(
                    scenario_dir,
                    scenario_name,
                    fps=float(getattr(args, 'video_fps', 10)),
                    resize_factor=int(getattr(args, 'video_resize_factor', 2)),
                )
                if video_paths:
                    print(f"[OK] Generated {len(video_paths)} videos")
                else:
                    print("[WARN] No videos generated (no image directories found)")
        else:
            print(f"[WARN] CARLA routes directory not found: {carla_routes_dir}")
            print("[WARN] Skipping run_custom_eval.")


def _run_carla_scenario(
    routes_dir: Path,
    port: int,
    planner: Optional[str] = None,
    overwrite: bool = True,
    actor_control_mode: str = "policy",
    walker_control_mode: str = "policy",
    capture_logreplay_images: bool = False,
    capture_every_sensor_frame: bool = False,
    npc_only_fake_ego: bool = False,
    scenario_name: Optional[str] = None,
) -> None:
    """Run the CARLA scenario using tools/run_custom_eval.py.
    
    When planner='log-replay', ego follows exact trajectory with timing.
    Actor control mode determines how NPCs behave:
      - 'policy': Use CARLA's AI planners (WaypointFollower)
      - 'replay': Use transform log replay for NPCs
    
    Args:
        routes_dir: Path to the routes directory
        port: CARLA server port
        planner: Planner type ('log-replay', 'autopilot', etc.)
        overwrite: Overwrite existing results
        actor_control_mode: 'policy' or 'replay' for NPCs
        walker_control_mode: 'policy' or 'replay' for walkers
        capture_logreplay_images: Save log-replay images
        capture_every_sensor_frame: Save all sensor frames
        npc_only_fake_ego: Run no-ego mode with ego trajectories replayed as custom NPC actors
        scenario_name: Name for results folder (defaults to routes_dir name)
    """
    repo_root = Path(__file__).resolve().parents[2]
    run_custom_eval_script = repo_root / "tools" / "run_custom_eval.py"
    
    if not run_custom_eval_script.exists():
        print(f"[ERROR] run_custom_eval.py not found: {run_custom_eval_script}")
        return
    
    # Use scenario_name for results folder, fall back to routes_dir name
    results_tag = scenario_name if scenario_name else routes_dir.name
    
    python_bin = sys.executable
    cmd = [
        python_bin,
        str(run_custom_eval_script),
        "--routes-dir", str(routes_dir),
        "--port", str(port),
        "--custom-actor-control-mode", actor_control_mode,
        "--results-tag", results_tag,
    ]
    
    if overwrite:
        cmd.append("--overwrite")
    
    # Always normalize actor z so walkers/pedestrians aren't spawned underground
    cmd.append("--normalize-actor-z")
    
    if planner:
        cmd.extend(["--planner", planner])
    if npc_only_fake_ego:
        cmd.append("--npc-only-fake-ego")
    
    # For replay mode on actors, enable log replay actors flag
    if actor_control_mode == "replay":
        cmd.append("--log-replay-actors")
    
    # Image capture flags - enable both for proper dense capture
    if capture_logreplay_images:
        cmd.append("--capture-logreplay-images")
        cmd.append("--capture-every-sensor-frame")  # Required for dense image capture
    elif capture_every_sensor_frame:
        cmd.append("--capture-every-sensor-frame")
    
    print(f"[INFO] Running CARLA scenario: {' '.join(cmd)}")
    print(f"[INFO]   Scenario name: {results_tag}")
    print(f"[INFO]   Planner: {planner or 'default'}")
    print(f"[INFO]   Actor control: {actor_control_mode}")
    print(f"[INFO]   Walker control: {walker_control_mode}")
    if npc_only_fake_ego:
        print("[INFO]   Mode: npc-only fake-ego")
    if capture_logreplay_images or capture_every_sensor_frame:
        print(f"[INFO]   Image capture: logreplay={capture_logreplay_images}, every_frame={capture_logreplay_images or capture_every_sensor_frame}")
    
    try:
        subprocess.run(cmd, check=True)
        print("[OK] CARLA scenario completed successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] run_custom_eval.py failed with exit code {exc.returncode}")
    except FileNotFoundError as exc:
        print(f"[ERROR] Could not run scenario: {exc}")


if __name__ == "__main__":
    main()
