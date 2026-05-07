"""
PerceptionSwapAgent — leaderboard agent that pairs a swappable HEAL detector
with a fixed rule-based planner. Multi-ego cooperative mode supported.

Purpose: measure planner sensitivity to perception quality. Same scenario,
same planner — only the detector backbone changes.

Configuration YAML (path_to_conf_file):

    detector: fcooper             # one of: fcooper, attfuse, disco
    cooperative: true             # true → fuse all egos in comm range
                                  # false → degenerate single-agent forward pass
    comm_range_m: 70              # default from HEAL configs (V2V comm distance)
    max_cooperators: 4            # cap fused agents (besides primary)
    cruise_speed: 6.0             # m/s
    score_threshold: 0.20         # detection confidence cutoff
    detection_radius: 2.0         # m, planner treats each det as a circle of this radius

For closed-loop CARLA via run_custom_eval.py: the harness applies the returned
controls directly. Per-ego perception runs once per tick (one cooperative
forward pass per ego, with itself as primary).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.perception_runner import PerceptionRunner
from team_code.rule_planner import RulePlanner, EgoState, Detection
from team_code.basic_agent_planner import BasicAgentPlanner
from team_code.behavior_agent_planner import BehaviorAgentPlanner
from team_code.infra_lidar_config import INFRA_LIDAR_POSES, is_infra_collab_enabled

# Default radius for the planner-side self-filter. A detection whose center
# is within this distance of the ego origin (in the ego's LiDAR frame) is
# dropped before it reaches the planner. Cooperative LiDAR fusion routinely
# produces a vehicle-sized phantom on top of the host ego (the partner ego's
# view of the host gets fused in), and some detectors (notably DiscoNet)
# also produce a self-return offset by ~half a vehicle length forward. A
# real lead vehicle's bbox center sits at >=4–5 m in bumper-to-bumper
# stops, so a radius up to ~3 m is safe to drop without masking real
# obstacles. Tunable per-run via the agent YAML key ``self_filter_radius_m``
# or the env override ``PERCEPTION_SWAP_SELF_FILTER_RADIUS_M``.
_PLANNER_SELF_FILTER_RADIUS_M = 3.0


def _parse_ego_replay_xml(xml_path: str):
    """Parse a ucla_v2_custom_ego_vehicle_<i>_REPLAY.xml into (times, xy).

    Returns (np.ndarray (T,), np.ndarray (T, 2)) sorted by time, or
    (None, None) if the file is missing/unparseable.
    """
    import xml.etree.ElementTree as ET
    if not xml_path or not os.path.isfile(xml_path):
        return None, None
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None, None
    ts, xs, ys = [], [], []
    for wp in root.iter("waypoint"):
        try:
            ts.append(float(wp.attrib["time"]))
            xs.append(float(wp.attrib["x"]))
            ys.append(float(wp.attrib["y"]))
        except (KeyError, ValueError):
            continue
    if not ts:
        return None, None
    order = np.argsort(ts)
    times = np.asarray(ts, dtype=np.float64)[order]
    xy = np.asarray(list(zip(xs, ys)), dtype=np.float64)[order]
    return times, xy


def _gt_xy_at(times, xy, t):
    """Linear interpolation of (times, xy) at scalar ``t``. Clamps to ends.

    Returns ``np.ndarray (2,)`` or ``None`` if data is missing.
    """
    if times is None or xy is None or len(times) == 0:
        return None
    if t <= times[0]:
        return xy[0].copy()
    if t >= times[-1]:
        return xy[-1].copy()
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = max(0, min(i, len(times) - 2))
    span = times[i + 1] - times[i]
    if span <= 1e-9:
        return xy[i].copy()
    frac = (t - times[i]) / span
    return xy[i] + frac * (xy[i + 1] - xy[i])


def _load_visible_ids_per_frame(agent_dir: str):
    """Parse all <frame>.yaml files in a v2x-real-pnp agent dir into
    ``{frame_idx: set[str]}`` of vehicle IDs that agent observed at that frame.

    The IDs come from the YAML's top-level ``vehicles:`` map. Keys are cast to
    str so they can be compared against CARLA ``actor.attributes['role_name']``
    (which is always a string regardless of how the manifest names were typed).

    Returns ``({}, 0)`` if the dir is missing or no frames parse.

    Avoids the PyYAML dependency (not all conda envs have it) — the
    ``vehicles:`` block is shallow enough that a hand-rolled scan over only the
    integer-keyed entries at the right indent works fine.
    """
    out = {}
    if not agent_dir or not os.path.isdir(agent_dir):
        return out, 0
    yaml_files = sorted(
        f for f in os.listdir(agent_dir)
        if f.endswith(".yaml") and f[:-5].isdigit()
    )
    parsed = 0
    for fn in yaml_files:
        try:
            frame_idx = int(fn[:-5])
        except ValueError:
            continue
        path = os.path.join(agent_dir, fn)
        ids = set()
        in_vehicles = False
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("vehicles:"):
                        in_vehicles = True
                        continue
                    if in_vehicles:
                        # End of the block when we hit another top-level key.
                        if line and not line.startswith((" ", "\t")) and line.strip():
                            break
                        # Keys are at exactly two leading spaces, integer + colon.
                        s = line.rstrip("\n")
                        if (len(s) > 3 and s.startswith("  ")
                                and not s.startswith("   ")
                                and s.endswith(":")):
                            key = s[2:-1].strip()
                            if key.isdigit():
                                ids.add(key)
        except Exception:
            continue
        out[frame_idx] = ids
        parsed += 1
    return out, parsed


# Datasets we know how to walk for the per-frame visible-actor filter. Search
# order matches the v2x-real-pnp layout: train > val > test > v2v_test.
_V2X_REAL_PNP_SPLITS = ("train", "val", "test", "v2v_test")


def _resolve_real_dataset_scene_dir(scene_basename: str,
                                    real_root: str) -> Optional[str]:
    """Return the v2x-real-pnp <split>/<scene> dir for ``scene_basename``,
    or None if the scene isn't present in any split."""
    if not scene_basename or not real_root or not os.path.isdir(real_root):
        return None
    for split in _V2X_REAL_PNP_SPLITS:
        cand = os.path.join(real_root, split, scene_basename)
        if os.path.isdir(cand):
            return cand
    return None


# ─── Env-var knobs for the perception-ablation experiment ───────────────────
# PERCEPTION_PLANNER:
#   "rule"        → original RulePlanner (default; preserves prior behaviour).
#   "basicagent"  → CARLA BasicAgent with hazard check rewired to our Detection
#                   list. This is the methodologically clean planner for the
#                   "perception is the only independent variable" study.
# PERCEPTION_USE_GT:
#   "1" / "true" → bypass the perception model and feed Detections derived from
#                  CARLA's world.get_actors() (range-gated to LiDAR sensor).
#                  This is the GT-oracle control row of the matrix.
def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "").lower() in ("1", "true", "yes", "on")


def get_entry_point():
    return "PerceptionSwapAgent"


class PerceptionSwapAgent(AutonomousAgent):
    """Per-ego perception+planner. v1: single ego only."""

    # ─────── Lifecycle ────────────────────────────────────────────────

    def setup(self, path_to_conf_file: str, ego_vehicles_num: int = 1):
        self.track = Track.SENSORS
        self.ego_vehicles_num = int(ego_vehicles_num)
        self.agent_name = "perception_swap"

        cfg = _load_conf(path_to_conf_file)
        self._detector_name    = str(cfg.get("detector", "fcooper"))
        # Tag used to prefix the agent's /tmp/* output files (diag JSONL + AP
        # summary). Defaults to detector_name so existing flows keep their
        # filenames; override with `log_tag: <unique>` in the YAML when two
        # configs would otherwise collide (e.g. perception_swap_gt and
        # perception_swap_fcooper both load detector=fcooper for the model
        # weights but should write to distinct logs in a parallel matrix).
        self._log_tag          = str(cfg.get("log_tag", self._detector_name))
        self._cooperative      = bool(cfg.get("cooperative", True))
        self._comm_range_m     = float(cfg.get("comm_range_m", 70.0))
        self._max_cooperators  = int(cfg.get("max_cooperators", 4))
        self._cruise_speed     = float(cfg.get("cruise_speed", 6.0))
        # Persistence filter: an in-corridor detection must hold for N
        # consecutive ticks before brake_now commits. Filters single-frame
        # perception FPs that previously locked the ego in dense walker
        # scenes. 1 disables the filter (legacy); 3 (~150 ms @ 20 Hz) is the
        # default and works for both RulePlanner and BasicAgentPlanner.
        self._brake_persistence_ticks = int(cfg.get("brake_persistence_ticks", 3))
        # Goal-reached latch params for RulePlanner. The latch fires when:
        #   (near_goal: cursor=last AND dist ≤ terminal_radius_m)  OR
        #   (past_end:  ego has crossed last waypoint in route's terminal dir)
        # AND ego has traveled ≥ min_progress_m from spawn (anti-spawn-trap;
        # without this gate, short routes like v2xpnp 13_1 (5m) lock at tick 0
        # because spawn is already within terminal_radius_m of the goal).
        self._terminal_radius_m = float(os.environ.get(
            "PERCEPTION_SWAP_TERMINAL_RADIUS_M",
            cfg.get("terminal_radius_m", 8.0),
        ))
        # Tight radius — safety latch when ego reaches goal exactly.
        # Default 0.5m (very tight); the PRIMARY latch is past_end so ego
        # actually crosses the route end (RC → 100%). The creep zone runs
        # at creep_speed_m_s for up to creep_timeout_s, after which we
        # force-latch so AgentBlockedTest doesn't kill us with a penalty.
        self._tight_terminal_radius_m = float(os.environ.get(
            "PERCEPTION_SWAP_TIGHT_TERMINAL_RADIUS_M",
            cfg.get("tight_terminal_radius_m", 0.5),
        ))
        self._creep_speed_m_s = float(os.environ.get(
            "PERCEPTION_SWAP_CREEP_SPEED_M_S",
            cfg.get("creep_speed_m_s", 1.5),
        ))
        self._creep_timeout_s = float(os.environ.get(
            "PERCEPTION_SWAP_CREEP_TIMEOUT_S",
            cfg.get("creep_timeout_s", 12.0),
        ))
        self._terminal_min_progress_m = float(os.environ.get(
            "PERCEPTION_SWAP_TERMINAL_MIN_PROGRESS_M",
            cfg.get("terminal_min_progress_m", 2.0),
        ))
        # Realistic vehicle half-width for the planner's "is in our lane?"
        # check. Default 0.9m (~typical sedan half-width). Replaces the old
        # behavior of using detection_radius (1.5m) for the lateral filter,
        # which made the effective corridor 6.2m wide and treated parked
        # cars 2-3m laterally as in-lane blockers.
        self._lateral_clearance_m = float(os.environ.get(
            "PERCEPTION_SWAP_LATERAL_CLEARANCE_M",
            cfg.get("lateral_clearance_m", 0.9),
        ))
        # Stationary-ego TTC bypass — when ego.speed < stopped_speed_m_s and
        # the gap to the obstacle exceeds stopped_min_gap_m, skip the brake
        # decision (real motion-based TTC at ego.speed=0 is ∞, no risk).
        # Without this, the rule planner has a permanent dead-stop trap
        # near static obstacles. Observed: cobevt 11_0 ego1 sat stationary
        # for 800+ ticks 6.6m away from a parked car.
        self._stopped_speed_m_s = float(os.environ.get(
            "PERCEPTION_SWAP_STOPPED_SPEED_M_S",
            cfg.get("stopped_speed_m_s", 0.3),
        ))
        # 0.5m bypass threshold — small enough to break the user's actual
        # observed deadlock (cobevt 11_0 ego1: gap=1.1m, ego stuck 800+ ticks)
        # but large enough that a genuinely touching obstacle still brakes.
        # Safety: the moment ego releases brake and accelerates above
        # stopped_speed_m_s, the next tick's TTC math takes over and re-brakes
        # if the obstacle is still in lane and approaching, so the worst-case
        # extra exposure is ~brake_persistence_ticks (150ms @ 20Hz) of creep.
        self._stopped_min_gap_m = float(os.environ.get(
            "PERCEPTION_SWAP_STOPPED_MIN_GAP_M",
            cfg.get("stopped_min_gap_m", 0.5),
        ))
        self._score_threshold  = float(cfg.get("score_threshold", 0.20))
        self._detection_radius = float(cfg.get("detection_radius", 2.0))
        # Self-filter radius applied to the planner detection list and the
        # BEV viz overlays. Env var wins over YAML so noisy detectors can be
        # tuned without touching their config files.
        self._self_filter_radius_m = float(os.environ.get(
            "PERCEPTION_SWAP_SELF_FILTER_RADIUS_M",
            cfg.get("self_filter_radius_m", _PLANNER_SELF_FILTER_RADIUS_M),
        ))
        self._collect_ap       = bool(cfg.get("collect_ap_metrics", True))
        # When disable_brake is True, we hand the planner an empty detection list
        # so it never triggers brake_now. Useful for AP-only runs where we want
        # the ego to keep moving through the scenario regardless of perception
        # noise. Independent of AP collection (which always sees true preds).
        self._disable_brake    = bool(cfg.get("disable_brake", False))

        # Planner selection + GT-perception toggle (see module-level docstring
        # for the experimental rationale).
        self._planner_kind     = os.environ.get(
            "PERCEPTION_PLANNER", str(cfg.get("planner", "rule"))
        ).lower().strip()
        if self._planner_kind not in ("rule", "basicagent", "behavior"):
            print(f"[perception_swap] unknown PERCEPTION_PLANNER={self._planner_kind!r}, falling back to 'rule'")
            self._planner_kind = "rule"
        self._use_gt_perception = _env_flag(
            "PERCEPTION_USE_GT", bool(cfg.get("use_gt_perception", False))
        )
        # Stored routes-per-ego, used to defer BasicAgent construction until
        # the hero actors have been spawned (the harness calls set_global_plan
        # before the first run_step, but actors only exist on/after run_step).
        self._stored_world_plans: List[Optional[list]] = [None] * self.ego_vehicles_num
        # Target speed for BasicAgent (km/h). Reuses cruise_speed (m/s) so the
        # config knob name stays consistent with the RulePlanner path.
        self._target_speed_kmh = float(cfg.get("target_speed_kmh", self._cruise_speed * 3.6))
        print(
            f"[perception_swap] planner={self._planner_kind} "
            f"use_gt_perception={self._use_gt_perception} "
            f"target_speed_kmh={self._target_speed_kmh:.1f} "
            f"disable_brake={self._disable_brake}"
        )
        if self._disable_brake:
            print(
                "[perception_swap] WARNING: disable_brake=True hides ALL detections "
                "from the planner. Closed-loop perception ablation is meaningless in "
                "this mode (every detector run, including GT, behaves identically — "
                "open-loop cruise). Set disable_brake=false in the YAML for the "
                "ablation matrix; keep it true ONLY for pure AP-collection runs."
            )

        # ─── Cooperative-perception robustness knobs (cross-model methodology) ───
        # Mirrors codriving / colmdriver / pnp_agent_e2e_v2v conventions so the
        # numbers from this agent are directly comparable to those models'
        # noise/latency tables.
        #
        # pose_error: Gaussian (or Laplace) noise added to each non-primary
        #   cooperator's [x, y, yaw] before pairwise transforms are computed
        #   (cf. opencood.utils.pose_utils.add_noise_data_dict). Primary's
        #   pose stays clean — equivalent to codriving's add_noise_data_dict
        #   (which noises everyone, but primary noise cancels in primary-frame
        #   AP eval). The realistic interpretation: cooperator GPS error.
        #
        # comm_latency: integer step delay applied to cooperator lidar+pose
        #   snapshots (cf. colmdriver_action.py:1326-1343). At 20 Hz CARLA
        #   tick rate, 6 steps = 300 ms, matching the inline comment in
        #   colmdriver_action. Primary always uses fresh data.
        #
        # YAML schema (both blocks optional → no perturbation when absent):
        #   pose_error:
        #     pos_std: 0.4   # meters, std of Gaussian on cooperator x/y
        #     rot_std: 0.4   # degrees, std of Gaussian on cooperator yaw
        #     pos_mean: 0.0
        #     rot_mean: 0.0
        #     use_laplace: false
        #   comm_latency: 6   # ticks (20 Hz → 300 ms)
        pose_err_cfg = cfg.get("pose_error") or {}
        if pose_err_cfg:
            self._pose_noise_active = True
            self._pose_noise_pos_std = float(pose_err_cfg.get("pos_std", 0.0))
            self._pose_noise_rot_std = float(pose_err_cfg.get("rot_std", 0.0))
            self._pose_noise_pos_mean = float(pose_err_cfg.get("pos_mean", 0.0))
            self._pose_noise_rot_mean = float(pose_err_cfg.get("rot_mean", 0.0))
            self._pose_noise_use_laplace = bool(pose_err_cfg.get("use_laplace", False))
            print(
                f"[perception_swap] pose_error ON: "
                f"{'Laplace' if self._pose_noise_use_laplace else 'Gaussian'} "
                f"pos(std={self._pose_noise_pos_std}, mean={self._pose_noise_pos_mean})  "
                f"rot(std={self._pose_noise_rot_std}, mean={self._pose_noise_rot_mean})"
            )
        else:
            self._pose_noise_active = False
            self._pose_noise_pos_std = 0.0
            self._pose_noise_rot_std = 0.0
            self._pose_noise_pos_mean = 0.0
            self._pose_noise_rot_mean = 0.0
            self._pose_noise_use_laplace = False

        self._comm_latency_steps = int(cfg.get("comm_latency", 0) or 0)
        if self._comm_latency_steps > 0:
            print(
                f"[perception_swap] comm_latency ON: {self._comm_latency_steps} ticks "
                f"(~{self._comm_latency_steps * 50} ms @ 20 Hz CARLA)"
            )
        # Ring buffer keyed by tick index → (lidars_copy, poses_copy). Used to
        # serve delayed cooperator data; primary data is always fresh. Trimmed
        # in run_step to keep at most latency+1 entries.
        self._latency_ring: Dict[int, Any] = {}

        # Single perception model loaded once. Reused across all egos and frames.
        self.perception = PerceptionRunner(
            detector=self._detector_name,
            score_threshold=self._score_threshold,
        )

        # One planner per ego — each ego has its own route + state.
        # Type is RulePlanner | BasicAgentPlanner depending on self._planner_kind.
        self.planners: List[Optional[Any]] = [None] * self.ego_vehicles_num

        # Per-ego visible-actor sets from the v2x-real-pnp source dataset:
        # for each frame in the source, ``{frame_idx: {role_name, ...}}``
        # listing which actors that ego actually saw at that timestamp.  When
        # populated AND the env flag is on, the AP collector uses this set
        # to restrict GT to those actors only — matching the real-world
        # observability instead of every CARLA vehicle in the scene.
        # Resolved lazily below from SAVE_PATH; ego_i → agent dir 1+i (the
        # mapping is verified empirically: ego_vehicle_0 ↔ agent dir "1",
        # ego_vehicle_1 ↔ agent dir "2", confirmed by aligning
        # ``true_ego_pose`` from the YAML to the ego's first XML waypoint
        # under the ucla_v2 map offset).
        self._visible_ids_per_frame: List[Dict[int, set]] = [
            {} for _ in range(self.ego_vehicles_num)
        ]
        self._gt_from_real_dataset = _env_flag(
            "PERCEPTION_SWAP_GT_FROM_REAL_DATASET", default=False,
        )
        # Hard-override that forces every visibility-filter source OFF.
        # Used by perception-quality correlation studies where AP must
        # count every CARLA vehicle in the eval range — not the dataset-
        # annotated subset and not the live semantic-LiDAR ≥N-hits subset.
        # Same env var is honored by openloop_helpers.OpenLoopAPHook so
        # codriving_* and perception_swap_* variants share methodology.
        self._disable_visibility_filter = _env_flag(
            "OPENLOOP_DISABLE_VISIBILITY_FILTER", default=False,
        )
        if self._disable_visibility_filter:
            self._gt_from_real_dataset = False
            print("[perception_swap] OPENLOOP_DISABLE_VISIBILITY_FILTER=1 — "
                  "visibility filter disabled (all CARLA vehicles in range count as GT)")
        # Source-dataset frame rate. v2x-real-pnp files at 10 Hz; the YAML
        # file index 000000..NNNNNN is the frame, indexed at this rate.
        self._real_dataset_fps = float(
            os.environ.get("PERCEPTION_SWAP_REAL_DATASET_FPS", "10.0")
        )

        # Cached LiDAR range from the model config — used by the GT projector.
        self._lidar_range = list(self.perception.config["heter"]["modality_setting"]["m1"]["preprocess"]["cav_lidar_range"])

        # AP eval region. The detector keeps its native cav_lidar_range
        # (typically 204.8 × 204.8 m) for the model forward pass, but AP
        # scoring + BEV display use a tighter crop centered on the planner-
        # relevant region. In openloop the default is 140 × 80 m
        # (x∈[-70,70], y∈[-40,40]); env vars allow ablation. Closed-loop is
        # untouched (still uses the full detector range). Logic is shared
        # with codriving / colmdriver via openloop_helpers so all three
        # agents score AP over the same AABB.
        try:
            from team_code.openloop_helpers import resolve_openloop_ap_eval_range
        except Exception:
            from openloop_helpers import resolve_openloop_ap_eval_range
        self._ap_eval_range = resolve_openloop_ap_eval_range(self._lidar_range)

        # One AP collector per ego (we measure detection quality per-ego since
        # each ego is the primary of its own cooperative forward pass).
        # Built AFTER the visible-id resolver below populates
        # self._visible_ids_per_frame; see end of setup().
        self.ap_collectors: List[Optional[CarlaGTApCollector]] = (
            [None] * self.ego_vehicles_num
        )

        # Will be filled with the CARLA actor IDs of the egos (so the GT
        # projector can exclude them). Populated lazily on first run_step.
        self._ego_actor_ids: Optional[set] = None

        # Where to write the AP summary JSON. Default to <SAVE_PATH>/perception_swap_ap.json
        # so each scene gets a durable, co-located file (mirrors the diag log).
        # Falls back to /tmp/perception_swap_ap_<tag>.json only when SAVE_PATH is
        # unset (probes outside of run_custom_eval). PERCEPTION_SWAP_AP_OUT env
        # remains an explicit override.
        _ap_save_root = os.environ.get("SAVE_PATH", "")
        if os.environ.get("PERCEPTION_SWAP_AP_OUT"):
            self._ap_out_path: Optional[str] = os.environ["PERCEPTION_SWAP_AP_OUT"]
        elif _ap_save_root:
            self._ap_out_path = os.path.join(_ap_save_root, "perception_swap_ap.json")
        else:
            self._ap_out_path = f"/tmp/perception_swap_ap_{self._log_tag}.json"

        # Diagnostics — last_step_info is keyed by ego index. Also write a JSONL
        # log of per-frame info so we can debug post-run.
        self.last_step_info: Dict[int, Dict[str, Any]] = {}
        # Default the diag path to <SAVE_PATH>/diag.jsonl so each scene's diag
        # is durable + co-located with its results. Falls back to /tmp/<tag>.jsonl
        # only when SAVE_PATH is unset (e.g. probes outside of run_custom_eval).
        # PERCEPTION_SWAP_DIAG_LOG (explicit env override) still wins. Read
        # SAVE_PATH locally here since the viz block below needs it too — keep
        # both reads using the same value to avoid divergence.
        _save_path_for_diag = os.environ.get("SAVE_PATH", "")
        if _save_path_for_diag:
            _diag_default = os.path.join(_save_path_for_diag, "perception_swap_diag.jsonl")
        else:
            _diag_default = f"/tmp/perception_swap_diag_{self._log_tag}.jsonl"
        diag_path = os.environ.get("PERCEPTION_SWAP_DIAG_LOG", _diag_default)
        try:
            os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
            self._diag_fp = open(diag_path, "w")
            print(f"[perception_swap] per-frame diag → {diag_path}")
        except Exception:
            self._diag_fp = None
        self._tick_count = 0

        # BEV viz: save a top-down PNG every N ticks showing route, predicted
        # 3s trajectory, predicted detections, and GT actors. Resolution order:
        #   1. PERCEPTION_SWAP_VIZ_DIR env (explicit override)
        #   2. SAVE_PATH env (run_custom_eval sets this to results/<tag>/image/<scene>)
        #      — matches the convention every other agent in this repo uses
        #   3. disabled
        viz_dir_override = os.environ.get("PERCEPTION_SWAP_VIZ_DIR", "")
        save_path_root  = os.environ.get("SAVE_PATH", "")
        if viz_dir_override:
            self._viz_dir = viz_dir_override
        elif save_path_root:
            self._viz_dir = os.path.join(save_path_root, "perception_swap_viz")
        else:
            self._viz_dir = ""
        self._viz_every = int(os.environ.get("PERCEPTION_SWAP_VIZ_EVERY", "10"))
        # Render the per-box AP overlay (TP/FP color, IoU + score labels) in
        # the BEV pane. On by default; set PERCEPTION_SWAP_BEV_AP_ANNOT=0 to
        # fall back to the simple scatter rendering.
        self._bev_ap_annot = os.environ.get(
            "PERCEPTION_SWAP_BEV_AP_ANNOT", "1"
        ).lower() in ("1", "true", "yes", "on")
        # IoU threshold used for TP/FP coloring in the BEV overlay. Matches the
        # standard cooperative-perception convention (also one of the AP
        # thresholds emitted to perception_swap_ap_*.json).
        try:
            self._bev_ap_iou_thresh = float(
                os.environ.get("PERCEPTION_SWAP_BEV_AP_IOU", "0.5")
            )
        except Exception:
            self._bev_ap_iou_thresh = 0.5
        # Whether to overlay GT boxes onto the front-camera annotated image.
        # Default OFF: GT boxes have a known frame mismatch (they're in actor
        # frame but the camera projection assumes lidar frame, so they appear
        # ~2.5 m too high in the image — which is why they used to look like
        # they were floating). The shift fix below handles this by subtracting
        # the lidar mount Z before projecting; opt in via this env var to see
        # GT on the camera.
        self._cam_gt_overlay = os.environ.get(
            "PERCEPTION_SWAP_CAM_GT_OVERLAY", "0"
        ).lower() in ("1", "true", "yes", "on")
        # Lidar mount height in actor frame (z component of the lidar sensor's
        # `z` attribute in `sensors()`). Used to convert actor-frame GT corners
        # back into lidar frame before camera projection. Keep in sync with the
        # sensor declaration if you ever change the lidar mount.
        self._lidar_mount_z = 2.5
        if self._viz_dir:
            try:
                os.makedirs(self._viz_dir, exist_ok=True)
                print(f"[perception_swap] BEV viz → {self._viz_dir} (every {self._viz_every} ticks)")
            except Exception as exc:
                print(f"[perception_swap] viz dir setup failed: {exc}")
                self._viz_dir = ""

        # Per-tick prediction-file output (JSON + NPZ) compatible with
        # carla_openloop_prototype's compute_metrics_from_run_dir reader.
        # Off by default — closed-loop baseline runs don't need it. Enable via
        # PERCEPTION_SWAP_WRITE_PRED_FILES=1; the openloop prototype's
        # run_planner_evaluation sets it automatically.
        self._write_pred_files = os.environ.get(
            "PERCEPTION_SWAP_WRITE_PRED_FILES", ""
        ).lower() in ("1", "true", "yes", "on")
        # Stride between prediction-file dumps. Codriving emits at 0.2s real
        # cadence (every 4 control ticks @ 20 Hz); we mirror that to give the
        # metric reader the same temporal density.
        self._pred_file_stride = int(os.environ.get("PERCEPTION_SWAP_PRED_FILE_STRIDE", "4"))
        if self._write_pred_files and save_path_root:
            self._pred_dirs = [
                os.path.join(save_path_root, f"ego_vehicle_{i}")
                for i in range(self.ego_vehicles_num)
            ]
            for d in self._pred_dirs:
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception as exc:
                    print(f"[perception_swap] pred-dir setup failed for {d}: {exc}")
            print(
                f"[perception_swap] per-tick prediction files → "
                f"<SAVE_PATH>/ego_vehicle_<i>/<step>.json + <step>_pred.npz "
                f"(stride={self._pred_file_stride})"
            )
        else:
            self._pred_dirs = []

        # Per-ego GT REPLAY trajectory for BEV viz overlays + per-tick ADE.
        # Resolution order: explicit env override → walk up SAVE_PATH parents
        # looking for a basename that matches a directory under
        # scenarioset/v2xpnp/. Skip silently if neither resolves — the BEV
        # viz then falls back to its existing behavior (no GT line, no
        # per-tick ADE).
        #
        # Path walk: SAVE_PATH typically ends in ``.../<planner>/<scene>/image``
        # so the scene-name component is the parent of the basename. We try
        # all ancestors (with ``_partial_<ts>`` retry suffix stripped) so
        # this remains correct if anyone restructures the results layout.
        self._gt_replays: List[Optional[tuple]] = [None] * self.ego_vehicles_num
        scenarioset_dir = os.environ.get("OPENLOOP_SCENARIOSET_DIR", "")
        if not scenarioset_dir and save_path_root:
            _repo = os.environ.get("OPENLOOP_REPO_ROOT", REPO_ROOT)
            _v2xpnp_root = os.path.join(_repo, "scenarioset", "v2xpnp")
            _path = os.path.abspath(save_path_root.rstrip("/"))
            for _ in range(6):  # cap walk depth
                _basename = os.path.basename(_path).split("_partial_", 1)[0]
                _candidate = os.path.join(_v2xpnp_root, _basename)
                if _basename and os.path.isdir(_candidate):
                    scenarioset_dir = _candidate
                    break
                _parent = os.path.dirname(_path)
                if _parent == _path:
                    break
                _path = _parent
        if scenarioset_dir:
            for i in range(self.ego_vehicles_num):
                xml = os.path.join(
                    scenarioset_dir,
                    f"ucla_v2_custom_ego_vehicle_{i}_REPLAY.xml",
                )
                t_arr, xy_arr = _parse_ego_replay_xml(xml)
                self._gt_replays[i] = (t_arr, xy_arr) if t_arr is not None else None
            n_loaded = sum(1 for r in self._gt_replays if r is not None)
            print(
                f"[perception_swap] BEV GT overlay: "
                f"{n_loaded}/{self.ego_vehicles_num} ego REPLAY loaded "
                f"from {scenarioset_dir}"
            )
        else:
            print(
                f"[perception_swap] BEV GT overlay disabled — "
                f"no scenarioset/v2xpnp/<scene> matched (SAVE_PATH={save_path_root})"
            )

        # Per-frame visible-actor sets from the v2x-real-pnp source dataset.
        # Keyed by frame index (0..N-1 in the source 10 Hz stream); each value
        # is the set of role_names that ego agent observed in that frame.
        # Used by CarlaGTApCollector to restrict GT to actors the real-world
        # ego *actually saw*, not every CARLA actor in the scene.
        scene_basename = os.path.basename(scenarioset_dir) if scenarioset_dir else ""
        real_root = os.environ.get(
            "PERCEPTION_SWAP_V2X_REAL_PNP_ROOT",
            "/data/dataset/v2x-real-pnp",
        )
        real_scene_dir = (
            _resolve_real_dataset_scene_dir(scene_basename, real_root)
            if scene_basename else None
        )
        if self._gt_from_real_dataset:
            if real_scene_dir is None:
                print(
                    f"[perception_swap] visible-actor GT filter requested but "
                    f"no v2x-real-pnp scene matched (scene={scene_basename!r}, "
                    f"root={real_root}); falling back to all-CARLA-vehicle GT."
                )
            else:
                # ego_vehicle_<i> ↔ agent dir str(i+1) in the source dataset.
                # Verified empirically: agent 1's true_ego_pose maps to
                # ego_0's spawn under the ucla_v2 map offset, agent 2 to
                # ego_1. Negative IDs in v2x-real-pnp are infrastructure
                # (RSU) sensors and do not correspond to runnable egos here.
                for i in range(self.ego_vehicles_num):
                    agent_id = str(i + 1)
                    agent_dir = os.path.join(real_scene_dir, agent_id)
                    visible, n_parsed = _load_visible_ids_per_frame(agent_dir)
                    self._visible_ids_per_frame[i] = visible
                    print(
                        f"[perception_swap] visible-actor filter ego{i}: "
                        f"agent dir {agent_dir!r} → {n_parsed} frames, "
                        f"avg visible={(sum(len(v) for v in visible.values())/max(1,len(visible))):.1f}"
                    )

        # Now build the AP collectors with the visible-id sets we just
        # resolved. Keep the rest of the call sites identical — collectors
        # without filtering get visible_ids_per_frame=None and behave as
        # before (every CARLA vehicle counts as GT).
        if self._collect_ap:
            self.ap_collectors = [
                CarlaGTApCollector(
                    visible_ids_per_frame=(
                        self._visible_ids_per_frame[i]
                        if (self._gt_from_real_dataset
                            and self._visible_ids_per_frame[i])
                        else None
                    )
                )
                for i in range(self.ego_vehicles_num)
            ]
        else:
            self.ap_collectors = [None] * self.ego_vehicles_num

        # Stationary infrastructure LiDARs (v2xpnp ablation). Spawned lazily
        # on the first run_step() iff COLMDRIVER_INFRA_COLLAB=1 AND the loaded
        # CARLA map is ucla_v2. _infra_actors is the list of spawned CARLA
        # sensor.lidar.ray_cast actors; _infra_buffers[i] holds the latest
        # full-sweep np.ndarray (N,4) in sensor-local frame for sensor i.
        self._infra_actors: list = []
        self._infra_buffers: list = []
        self._infra_spawn_attempted = False

    def setup_infra_sensors(self):
        """Public hook called by agent_wrapper.setup_sensors() before its
        priming world.tick() — guarantees infra has data at step 0.
        """
        self._ensure_infra_sensors()

    def _ensure_infra_sensors(self):
        """Lazy one-shot spawn of stationary infra LiDARs for v2xpnp scenes.

        No-op unless COLMDRIVER_INFRA_COLLAB=1 (set by --infra-collab) AND
        the loaded CARLA map is ucla_v2. Spawned LiDAR config matches the
        ego LiDAR (64 channels, 100m range, 20 Hz, 1.3M pts/s) so points
        are in-distribution for the perception model.
        """
        if self._infra_spawn_attempted:
            return
        self._infra_spawn_attempted = True
        if not is_infra_collab_enabled():
            return
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world = CarlaDataProvider.get_world()
            map_name = world.get_map().name if world is not None else ""
        except Exception:
            return
        if "ucla_v2" not in map_name.lower():
            print(f"[infra-collab] map '{map_name}' is not ucla_v2 — skipping infra LiDAR spawn")
            return
        bp_lib = world.get_blueprint_library()
        bp = bp_lib.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", "64")
        bp.set_attribute("range", "100.0")
        bp.set_attribute("rotation_frequency", "20.0")
        bp.set_attribute("points_per_second", "1300000")
        bp.set_attribute("upper_fov", "5.0")
        bp.set_attribute("lower_fov", "-25.0")
        bp.set_attribute("horizontal_fov", "360.0")
        carla_map = world.get_map()
        for i, p in enumerate(INFRA_LIDAR_POSES):
            # Configured z is height above ground in v2x-real-pnp world coords
            # (~0). At UCLA Westwood in CARLA, ground sits at ~10.7m absolute
            # world Z, so we add ground_z + p.z to keep the sensor above
            # ground (sub-ground spawn corrupts cooperative fusion → bad AP).
            ground_z = float(p.z)
            try:
                wp = carla_map.get_waypoint(
                    carla.Location(x=p.x, y=p.y, z=0.0),
                    project_to_road=True,
                    lane_type=carla.LaneType.Any,
                )
                if wp is not None:
                    ground_z = float(wp.transform.location.z) + float(p.z)
            except Exception:
                pass
            tf = carla.Transform(
                carla.Location(x=p.x, y=p.y, z=ground_z),
                carla.Rotation(roll=p.roll, pitch=p.pitch, yaw=p.yaw),
            )
            actor = world.spawn_actor(bp, tf)
            self._infra_actors.append(actor)
            # Buffer holds the latest full-sweep (N,4) sensor-local array.
            buf_slot = [None]
            self._infra_buffers.append(buf_slot)

            def _make_cb(slot=buf_slot):
                def _cb(data):
                    # np.frombuffer returns a READ-ONLY view of CARLA's raw
                    # buffer. Downstream voxelizers (spconv) modify the array
                    # in-place and crash with "array is not writeable" if we
                    # pass the view through. .copy() makes a writeable owned
                    # array, costs ~1 µs per sweep.
                    raw = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4).copy()
                    slot[0] = raw
                return _cb

            actor.listen(_make_cb())
            print(f"[infra-collab] spawned {p.name} at "
                  f"({p.x:.2f}, {p.y:.2f}, {ground_z:.2f}) yaw={p.yaw:.2f}° "
                  f"(config_z_above_ground={p.z:.2f}m)")

    def _collect_infra_lidars(self):
        """Return (lidar_arrays, world_poses) for spawned infra LiDARs.

        Lists are aligned: lidar_arrays[i] is an (N,4) sensor-local array
        and world_poses[i] is [x, y, z, roll, yaw, pitch] (deg). Skips
        sensors whose buffer hasn't received its first sweep yet.
        """
        if not self._infra_actors:
            return [], []
        lidars: list = []
        poses: list = []
        for actor, slot, p in zip(self._infra_actors, self._infra_buffers, INFRA_LIDAR_POSES):
            pts = slot[0]
            if pts is None or pts.shape[0] == 0:
                continue
            lidars.append(np.asarray(pts, dtype=np.float32))
            poses.append([
                float(p.x), float(p.y), float(p.z),
                float(p.roll), float(p.yaw), float(p.pitch),
            ])
        return lidars, poses

    def _cleanup_infra_sensors(self):
        for actor in self._infra_actors:
            try:
                actor.stop()
                actor.destroy()
            except Exception:
                pass
        self._infra_actors = []
        self._infra_buffers = []

    def destroy(self):
        # Defensive snapshot at the START of destroy() — if anything in the
        # cleanup chain below hard-exits the process (e.g. scenario_runner's
        # remove_all_actors hitting its 30 s budget on dense scenes), at least
        # we have the latest AP state on disk. The periodic per-tick snapshot
        # also covers this, but does so at 10-tick granularity; this catches
        # the final 1-9 ticks that wouldn't otherwise be flushed.
        try:
            self._snapshot_ap_summary()
        except Exception:
            pass

        # Tear down infrastructure LiDAR sensors before any heavier teardown
        # so they don't get orphaned in CARLA if a later step in destroy()
        # hard-exits.
        try:
            self._cleanup_infra_sensors()
        except Exception:
            pass

        # Close diag log.
        try:
            if getattr(self, "_diag_fp", None) is not None:
                self._diag_fp.close()
        except Exception:
            pass

        # Save AP summary, if collected.
        if self._collect_ap and any(c is not None for c in self.ap_collectors):
            out_path = self._ap_out_path
            for ego_i, coll in enumerate(self.ap_collectors):
                if coll is None:
                    continue
                p = out_path.replace(".json", f"_ego{ego_i}.json")
                try:
                    coll.save(p, extra_meta={
                        "detector":     self._detector_name,
                        "cooperative":  self._cooperative,
                        "comm_range_m": self._comm_range_m,
                        "ego_index":    ego_i,
                    })
                    print(f"[perception_swap] saved AP summary → {p}")
                except Exception as exc:
                    print(f"[perception_swap] AP save failed for ego {ego_i}: {exc}")

        # Free CUDA memory if torch sets up persistent caches.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _snapshot_ap_summary(self):
        """Write the current AP summary to disk so a kill mid-run doesn't lose data."""
        if not self._collect_ap:
            return
        out_path = self._ap_out_path
        for ego_i, coll in enumerate(self.ap_collectors):
            if coll is None:
                continue
            p = out_path.replace(".json", f"_ego{ego_i}.json")
            try:
                coll.save(p, extra_meta={
                    "detector":     self._detector_name,
                    "cooperative":  self._cooperative,
                    "comm_range_m": self._comm_range_m,
                    "ego_index":    ego_i,
                    "snapshot_tick": self._tick_count,
                    "is_snapshot":   True,
                })
            except Exception:
                pass

    def _save_per_step_pred_files(
        self, primary_idx, ego, det, traj, ego_pose_world,
    ):
        """Per-tick JSON + NPZ in the format carla_openloop_prototype.compute_metrics_from_run_dir
        consumes (codriving-compatible).

        JSON fields: lidar_pose_x/y, theta (≈ yaw + π/2), waypoints (10 ego-local
        waypoints at 0.2s cadence), speed.
        NPZ fields:  pred_box_world (N, 8, 3) world-frame BB corners,
                     pred_score (N,), lidar_pose_x/y, theta, ego_x, ego_y.

        Coord conventions copied from carla_openloop_prototype.local_waypoints_to_world:
            carla-body forward = -local_y
            carla-body left    = -local_x
        i.e. inverse:
            forward = cos(yaw)·(wx-lx) + sin(yaw)·(wy-ly)
            left    = -sin(yaw)·(wx-lx) + cos(yaw)·(wy-ly)
            local_y = -forward,  local_x = -left
        """
        if not self._write_pred_files or not self._pred_dirs:
            return
        # First-frame-only mode (the default for openloop runs now): dump the
        # pred + gt + planner waypoints snapshot at tick 0 and never again.
        # Closed-loop driving continues normally for DS/RC/CR scoring; ADE,
        # AP@.50 and the open-loop CR forecast are computed offline from
        # tick 0 by the existing recompute_ap_riskm and openloop_post_metrics
        # scripts (they iterate whatever pred/gt files exist on disk).
        # Set CUSTOM_OPENLOOP_PER_TICK_DUMP=1 to fall back to the legacy
        # per-stride dump for full-trajectory evaluation.
        per_tick = os.environ.get("CUSTOM_OPENLOOP_PER_TICK_DUMP", "0") == "1"
        if per_tick:
            if self._tick_count % self._pred_file_stride != 0:
                return
        else:
            if self._tick_count != 0:
                return
        try:
            from tools.perception_runner import Detections as _Det  # noqa: F401  (type cue)
            _, _, _, _, x_to_world = _lazy_eval_imports()

            # Ego world position + yaw.
            lx = float(ego_pose_world[0])
            ly = float(ego_pose_world[1])
            yaw_deg = float(ego_pose_world[4])
            yaw_rad = float(np.radians(yaw_deg))
            theta = yaw_rad + (np.pi / 2.0)   # codriving's compass convention

            # Sub-sample our 30-pt @ 0.1s trajectory to 10-pt @ 0.2s to match the
            # metric reader's pred_horizon_steps=10 / pred_dt_s=0.2 default.
            traj = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
            stride = max(1, int(round(0.2 / 0.1)))
            sub = traj[stride - 1::stride][:10]   # 10 waypoints @ 0.2s
            if len(sub) < 10:
                # Pad with last waypoint if trajectory is shorter than 2 s.
                pad = np.tile(sub[-1:] if len(sub) else np.zeros((1, 2)), (10 - len(sub), 1))
                sub = np.concatenate([sub, pad], axis=0)

            # World → ego-local in codriving convention.
            dx = sub[:, 0] - lx
            dy = sub[:, 1] - ly
            c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
            forward = c * dx + s * dy
            left    = -s * dx + c * dy
            local_y = -forward
            local_x = -left
            local_wps = np.column_stack([local_x, local_y]).tolist()

            # Predicted boxes in ego frame → world frame.
            T_world_ego = x_to_world(ego_pose_world)
            pred_corners_ego = np.asarray(det.corners, dtype=np.float64)   # (N, 8, 3)
            if pred_corners_ego.size:
                hom = np.concatenate(
                    [pred_corners_ego.reshape(-1, 3),
                     np.ones((pred_corners_ego.shape[0] * 8, 1))],
                    axis=1,
                )
                pred_world = (T_world_ego @ hom.T).T[:, :3].reshape(-1, 8, 3).astype(np.float32)
            else:
                pred_world = np.zeros((0, 8, 3), dtype=np.float32)
            pred_score = np.asarray(det.scores, dtype=np.float32).reshape(-1)

            # Output files.
            out_dir = self._pred_dirs[primary_idx]
            stem = f"{self._tick_count:04d}"
            json_path = os.path.join(out_dir, f"{stem}.json")
            npz_path  = os.path.join(out_dir, f"{stem}_pred.npz")

            payload = {
                "speed":          float(ego.speed),
                "waypoints":      local_wps,
                "lidar_pose_x":   lx,
                "lidar_pose_y":   ly,
                "theta":          float(theta),
                "yaw_rad":        yaw_rad,    # explicit, in addition to compass theta
            }
            with open(json_path, "w") as f:
                json.dump(payload, f)
            np.savez(
                npz_path,
                pred_box_world=pred_world,
                pred_score=pred_score,
                lidar_pose_x=np.float64(lx),
                lidar_pose_y=np.float64(ly),
                theta=np.float64(theta),
                ego_x=np.float64(lx),
                ego_y=np.float64(ly),
            )
            # Companion GT cache: every CARLA vehicle (sans egos) in world frame,
            # with per-actor metadata. Lets downstream AP recompute use the
            # exact GT the agent saw — no XML reparsing, no waypoint
            # interpolation guesses, no bbox.location quibbles. Only writes if
            # an AP collector exists for this ego (ensures _build_gt_world_with_meta
            # is wired). Skipping costs nothing for runs where collection is off.
            coll = self.ap_collectors[primary_idx] if primary_idx < len(self.ap_collectors) else None
            if coll is not None:
                if self._ego_actor_ids is None:
                    try:
                        self._ego_actor_ids = self._resolve_ego_actor_ids()
                    except Exception:
                        self._ego_actor_ids = set()
                # Visibility-filtered GT (OPV2V protocol): pass the set of
                # actor IDs that this ego's semantic LiDAR hit ≥N times this
                # tick. Falls through to None when semantic LiDAR isn't
                # declared (visibility filter disabled), giving the strict
                # HEAL all-in-range behavior.
                # OPENLOOP_DISABLE_VISIBILITY_FILTER=1 forces None here too,
                # so the cached _gt.npz also reflects the no-filter scope.
                vis_ids = (self._visible_ids_per_slot or {}).get(primary_idx) \
                    if hasattr(self, "_visible_ids_per_slot") else None
                if getattr(self, "_disable_visibility_filter", False):
                    vis_ids = None
                gt_w, gt_ids, gt_roles, gt_types = coll._build_gt_world_with_meta(
                    self._ego_actor_ids or set(),
                    visible_actor_ids=vis_ids,
                )
                gt_npz_path = os.path.join(out_dir, f"{stem}_gt.npz")
                np.savez(
                    gt_npz_path,
                    gt_box_world=gt_w,
                    gt_actor_ids=gt_ids,
                    gt_role_names=np.array(gt_roles, dtype=object),
                    gt_type_ids=np.array(gt_types, dtype=object),
                    ego_x=np.float64(lx),
                    ego_y=np.float64(ly),
                    theta=np.float64(theta),
                    visibility_filter_hits=int(os.environ.get(
                        "PERCEPTION_SWAP_VISIBILITY_FILTER_HITS", "10")),
                    visibility_filter_applied=bool(vis_ids is not None),
                )
        except Exception as exc:
            self._pred_file_err_count = getattr(self, "_pred_file_err_count", 0) + 1
            if self._pred_file_err_count <= 3:
                print(f"[perception_swap] pred-file save failed (ego {primary_idx}): {exc}")

    def _save_camera_frame(self, primary_idx, cam_img):
        """Write the raw front-camera RGB to a PNG, one per save-tick."""
        if not self._viz_dir:
            return
        if self._tick_count % self._viz_every != 0:
            return
        if cam_img is None or not hasattr(cam_img, "shape") or cam_img.size == 0:
            return
        try:
            # CARLA gives us BGRA uint8. Drop alpha + reorder to RGB for PIL.
            if cam_img.ndim == 3 and cam_img.shape[-1] == 4:
                rgb = cam_img[..., [2, 1, 0]]
            else:
                rgb = cam_img
            from PIL import Image
            out = os.path.join(
                self._viz_dir,
                f"tick_{self._tick_count:05d}_ego{primary_idx}_cam.png",
            )
            Image.fromarray(rgb.astype(np.uint8)).save(out)
        except Exception as exc:
            self._cam_err_count = getattr(self, "_cam_err_count", 0) + 1
            if self._cam_err_count <= 3:
                print(f"[perception_swap] camera save failed: {exc}")

    # Camera intrinsics derived once from the sensor declaration. Front camera:
    # x=1.3, y=0.0, z=1.5 (m, ego-relative); 800×600; FOV=100°. The LiDAR sits
    # at (0, 0, 2.5) ego-relative, so a point in LiDAR frame (where `det.corners`
    # live) maps to camera frame by translation only (both at yaw=0):
    #     cam_x = x_lidar − 1.3            (forward)
    #     cam_y = y_lidar                   (right)
    #     cam_z = z_lidar + 1.0             (up;  2.5 − 1.5)
    _CAM_W, _CAM_H, _CAM_FOV_DEG = 800, 600, 100.0
    _CAM_OFFSET_FROM_LIDAR = np.array([1.3, 0.0, -1.0], dtype=np.float64)  # subtract from lidar xyz

    # Top-down ("rgb_top") camera. Pitch=-90, attached to ego at z=60 m, FOV 90°.
    # Visible ground square is 2 * Z * tan(FOV/2) ≈ 120 × 120 m. Used as the
    # BEV background in the viz so we render the actual CARLA scene under the
    # TP/FP boxes instead of a synthetic matplotlib plot.
    _TOP_W, _TOP_H, _TOP_FOV_DEG, _TOP_Z = 800, 800, 90.0, 60.0

    @classmethod
    def _camera_project(cls, points_lidar):
        """Project (N, 3) lidar-frame points to (N, 2) image-plane pixel coords.
        Returns (uv, mask) where mask is True for points strictly in front of the camera.
        """
        pts = np.asarray(points_lidar, dtype=np.float64).reshape(-1, 3)
        cam = pts.copy()
        cam[:, 0] -= cls._CAM_OFFSET_FROM_LIDAR[0]   # forward
        cam[:, 1] -= cls._CAM_OFFSET_FROM_LIDAR[1]   # right
        cam[:, 2] -= cls._CAM_OFFSET_FROM_LIDAR[2]   # up (note minus → camera_z = lidar_z + 1.0)

        fx = cls._CAM_W / (2.0 * np.tan(np.radians(cls._CAM_FOV_DEG) / 2.0))
        fy = fx
        cx = cls._CAM_W / 2.0
        cy = cls._CAM_H / 2.0

        in_front = cam[:, 0] > 0.5
        u = np.full(len(cam), np.nan)
        v = np.full(len(cam), np.nan)
        ok = in_front
        u[ok] = fx * cam[ok, 1] / cam[ok, 0] + cx
        v[ok] = fy * (-cam[ok, 2]) / cam[ok, 0] + cy
        uv = np.column_stack([u, v])
        return uv, in_front

    def _save_camera_frame_annotated(self, primary_idx, cam_img, det, ego_pose_world):
        """Write the front-camera RGB with predicted 3D bounding boxes overlaid.

        Annotation is *purely post-hoc PIL drawing* on the saved PNG — boxes are
        never injected into the CARLA world. Other actors' sensors cannot
        observe these overlays.
        """
        if not self._viz_dir:
            return
        if self._tick_count % self._viz_every != 0:
            return
        if cam_img is None or not hasattr(cam_img, "shape") or cam_img.size == 0:
            return
        try:
            from PIL import Image, ImageDraw, ImageFont

            # CARLA → PIL.
            if cam_img.ndim == 3 and cam_img.shape[-1] == 4:
                rgb = cam_img[..., [2, 1, 0]].astype(np.uint8)
            else:
                rgb = cam_img.astype(np.uint8)
            img = Image.fromarray(rgb).copy()
            draw = ImageDraw.Draw(img, "RGBA")

            # 8-corner connectivity for our CCW-bottom convention:
            #   bottom face: 0-1, 1-2, 2-3, 3-0
            #   top face:    4-5, 5-6, 6-7, 7-4
            #   verticals:   0-4, 1-5, 2-6, 3-7
            edges = [(0,1),(1,2),(2,3),(3,0),
                     (4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]

            # ----- Per-box AP match for color coding -----
            # Same matcher the BEV uses. Returns (gt_corners_ego, pred_status,
            # pred_iou, gt_matched, gt_visible). When None (annotation off,
            # GT-oracle mode, or shapely missing) we fall back to legacy
            # yellow boxes for predictions only.
            match = None
            if self._bev_ap_annot:
                match = self._compute_per_box_ap_match(
                    primary_idx, det, ego_pose_world,
                    iou_thresh=self._bev_ap_iou_thresh,
                )
            gt_corners_ego = None
            pred_status: list = []
            pred_iou: list = []
            gt_matched: list = []
            gt_visible: list = []
            if match is not None:
                gt_corners_ego, pred_status, pred_iou, gt_matched, gt_visible, gt_hits = match

            # Color palette (RGBA). High alpha so wireframes stay readable
            # against busy CARLA backgrounds.
            COL_TP_PRED       = (40, 220, 80, 235)    # bright green
            COL_FP_PRED       = (235, 50, 50, 235)    # bright red
            COL_GT_MATCHED    = (40, 160, 60, 200)    # dim green
            COL_GT_MISSED     = (200, 200, 200, 220)  # neutral gray
            COL_GT_NONVIS     = (90, 140, 230, 170)   # blue (filtered-out by visibility)
            COL_LEGACY_PRED   = (255, 230, 0, 230)    # yellow (no-match fallback)

            def _draw_box_3d(uv, color, width=2):
                """Draw the 12 wireframe edges of a projected 3D box."""
                for a, b in edges:
                    draw.line(
                        [(float(uv[a, 0]), float(uv[a, 1])),
                         (float(uv[b, 0]), float(uv[b, 1]))],
                        fill=color, width=width,
                    )

            def _box_visible(uv, in_front):
                """Cheap visibility filter: all 8 corners in front of the
                near plane AND box bounds intersect the image rect."""
                if not bool(np.all(in_front)):
                    return False
                u_arr, v_arr = uv[:, 0], uv[:, 1]
                if (u_arr.max() < 0 or u_arr.min() > self._CAM_W
                        or v_arr.max() < 0 or v_arr.min() > self._CAM_H):
                    return False
                return True

            # ----- 1. GT boxes (drawn first so predictions overlay on top) -----
            # Off by default — set PERCEPTION_SWAP_CAM_GT_OVERLAY=1 to enable.
            # Frame conversion fix: GT corners come back from
            # `_compute_per_box_ap_match` in ACTOR frame (because
            # primary_pose_world is the actor's transform, not the lidar's),
            # but `_camera_project` assumes its input is in LIDAR frame.
            # We bridge the two by subtracting the lidar mount height (Z=2.5)
            # from the actor-frame Z before projection. Without this, GT
            # boxes appear ~2.5 m too high in the image and look like they're
            # floating above the vehicles.
            n_gt_drawn = 0
            if self._cam_gt_overlay and gt_corners_ego is not None:
                for gi in range(len(gt_corners_ego)):
                    box_actor = np.asarray(gt_corners_ego[gi], dtype=np.float64).copy()
                    box_actor[:, 2] -= self._lidar_mount_z   # actor → lidar frame
                    uv, in_front = self._camera_project(box_actor)
                    if not _box_visible(uv, in_front):
                        continue
                    matched = bool(gt_matched[gi]) if gi < len(gt_matched) else False
                    is_vis = bool(gt_visible[gi]) if gi < len(gt_visible) else True
                    if not is_vis:
                        color = COL_GT_NONVIS
                    elif matched:
                        color = COL_GT_MATCHED
                    else:
                        color = COL_GT_MISSED
                    # Thinner line + dimmer color so prediction boxes overlaid
                    # next stay the visually-dominant element.
                    _draw_box_3d(uv, color, width=1)
                    n_gt_drawn += 1

            # ----- 2. Predicted boxes -----
            corners = np.asarray(det.corners, dtype=np.float64)   # (P, 8, 3)
            scores  = np.asarray(det.scores, dtype=np.float64).reshape(-1)
            n_drawn = 0
            n_tp = n_fp = 0
            for i in range(corners.shape[0]):
                box_lidar = corners[i]                            # (8, 3)
                uv, in_front = self._camera_project(box_lidar)
                if not _box_visible(uv, in_front):
                    continue
                if pred_status:
                    status = pred_status[i] if i < len(pred_status) else "FP"
                    if status == "OOR":
                        # OOR: out of AP eval range — draw faintly, don't count.
                        _draw_box_3d(uv, (255, 165, 82, 140), width=1)
                        continue
                    is_tp = status == "TP"
                    color = COL_TP_PRED if is_tp else COL_FP_PRED
                    iou_val = pred_iou[i] if i < len(pred_iou) else 0.0
                    label = (
                        f"{float(scores[i]):.2f}|IoU{iou_val:.2f}"
                        if is_tp else f"{float(scores[i]):.2f}"
                    )
                    if is_tp:
                        n_tp += 1
                    else:
                        n_fp += 1
                else:
                    color = COL_LEGACY_PRED
                    label = f"{float(scores[i]):.2f}"
                _draw_box_3d(uv, color, width=2)
                # Score / IoU label at the top-left of the box's image bbox.
                u, v = uv[:, 0], uv[:, 1]
                tl = (float(np.nanmin(u)), float(np.nanmin(v)) - 12)
                draw.text(tl, label, fill=color)
                n_drawn += 1

            # Total FN comes from the matcher's gt_matched array, not from
            # what we drew. When GT overlay is off we still want to surface
            # the real FN count in the banner so the AP picture is honest.
            n_fn_total = (
                int(sum(1 for m in gt_matched if not m))
                if pred_status else 0
            )

            # Header banner — confirms what's overlaid + tick count.
            if pred_status:
                ap_str = (
                    f"  TP={n_tp} FP={n_fp} FN={n_fn_total} "
                    f"@IoU{self._bev_ap_iou_thresh:.2f}"
                )
            else:
                ap_str = ""
            gt_str = (
                f"  GT(drawn)={n_gt_drawn}" if self._cam_gt_overlay
                else "  GT(overlay=off)"
            )
            banner = (
                f"tick {self._tick_count:05d}  ego_{primary_idx}  "
                f"detector={self._detector_name}  pred={n_drawn}/{corners.shape[0]}  "
                f"{gt_str}  GT-mode={self._use_gt_perception}{ap_str}"
            )
            draw.rectangle([(0, 0), (self._CAM_W, 22)], fill=(0, 0, 0, 180))
            draw.text((6, 4), banner, fill=(255, 255, 255, 255))

            out = os.path.join(
                self._viz_dir,
                f"tick_{self._tick_count:05d}_ego{primary_idx}_cam_annot.png",
            )
            img.save(out)
        except Exception as exc:
            self._cam_annot_err_count = getattr(self, "_cam_annot_err_count", 0) + 1
            if self._cam_annot_err_count <= 3:
                print(f"[perception_swap] annotated camera save failed: {exc}")

    def _compute_per_box_ap_match(self, primary_idx, det, ego_pose_world,
                                  iou_thresh: float = 0.5):
        """Match each predicted box to ground-truth via BEV polygon IoU.

        Returns ``(gt_corners_ego, pred_status, pred_iou, gt_matched, gt_visible, gt_hits)``
        where
            gt_corners_ego : (M, 8, 3) ndarray of GT box corners in ego frame
            pred_status    : list[str] length P, "TP" or "FP" at iou_thresh
            pred_iou       : list[float] length P, best IoU per pred (regardless of TP/FP)
            gt_matched     : list[bool]  length M, True iff some pred TP'd it
            gt_visible     : list[bool]  length M, True iff GT actor is in the
                             semantic-LiDAR visibility set (or True for all
                             when no visibility filter is active)
            gt_hits        : list[int]   length M, semantic-LiDAR ray hits per
                             GT actor (0 when no semantic-LiDAR data was read)

        Only visible GT boxes participate in matching — visible-but-unmatched
        boxes are FN, and a pred that overlaps only a non-visible GT becomes
        FP (matches the OPV2V visibility-filtered AP protocol). Non-visible
        boxes are still returned so the BEV viz can show them in a third
        color for context.

        Returns ``None`` if shapely is unavailable, GT extraction fails, or
        we're rendering for the GT-oracle variant (preds == GT, AP trivially 1).
        """
        try:
            from shapely.geometry import Polygon
        except Exception:
            return None

        # GT-oracle mode: predictions ARE GT. AP is trivially perfect.
        # Returning None lets the caller fall back to the simple renderer
        # (no point coloring something that's TP by definition).
        if self._use_gt_perception:
            return None

        coll = self.ap_collectors[primary_idx] if primary_idx < len(self.ap_collectors) else None
        if coll is None:
            return None
        if self._ego_actor_ids is None:
            try:
                self._ego_actor_ids = self._resolve_ego_actor_ids()
            except Exception:
                return None
        # Pull the unfiltered GT set (with actor IDs) so we can render visible
        # vs non-visible separately. We do NOT pass visible_actor_ids here —
        # filtering happens locally below so non-visible boxes survive for viz.
        # Uses _ap_eval_range (cropped scoring region) to match what AP scores.
        try:
            gt_corners_ego, gt_actor_ids = coll._build_gt_corners_ego_frame(
                primary_pose_world=ego_pose_world,
                lidar_range=self._ap_eval_range,
                ego_actor_ids=self._ego_actor_ids,
                return_actor_ids=True,
            )
        except Exception:
            return None

        # Visibility set (semantic-LiDAR derived). May be None when the
        # filter is disabled, in which case every GT counts as "visible".
        vis_actor_ids = (
            (self._visible_ids_per_slot or {}).get(primary_idx)
            if hasattr(self, "_visible_ids_per_slot") else None
        )
        # Per-actor hit counts (so BEV can label each GT box with its count).
        hits_map = (
            (self._hits_per_actor_per_slot or {}).get(primary_idx, {})
            if hasattr(self, "_hits_per_actor_per_slot") else {}
        )
        gt_hits = [int(hits_map.get(int(aid), 0)) for aid in gt_actor_ids.tolist()]
        if vis_actor_ids is None:
            gt_visible = [True] * int(len(gt_corners_ego))
        else:
            gt_visible = [int(aid) in vis_actor_ids for aid in gt_actor_ids.tolist()]

        P = int(det.P)
        M = int(len(gt_corners_ego))
        pred_status = ["FP"] * P
        pred_iou = [0.0] * P
        gt_matched = [False] * M
        if P == 0 or M == 0:
            return gt_corners_ego, pred_status, pred_iou, gt_matched, gt_visible, gt_hits

        # Build BEV polygons. The 8x3 corners use the bottom-4 / top-4 layout
        # from _build_gt_corners_ego_frame; the bottom face is corners[:4]
        # in CCW order. Same convention for predictions per the cooperative
        # perception models (see boxes_to_corners_3d).
        def _poly(box):
            try:
                return Polygon([(float(box[i][0]), float(box[i][1])) for i in range(4)])
            except Exception:
                return None

        pred_polys = [_poly(b) for b in det.corners]
        gt_polys = [_poly(b) for b in gt_corners_ego]

        # AP eval range filter for predictions: a pred box whose center lies
        # outside the cropped scoring region gets tagged "OOR" — skipped from
        # matching, never counted as TP or FP. Mirrors what AP collector does
        # at scoring time; without this, every OOR pred would be a spurious FP.
        ap_lr = self._ap_eval_range
        ap_xmn, ap_ymn, ap_xmx, ap_ymx = ap_lr[0], ap_lr[1], ap_lr[3], ap_lr[4]

        # Greedy assignment by descending pred score (the standard AP recipe).
        # Only visible GTs participate in matching: a pred that overlaps only
        # a non-visible GT becomes FP (OPV2V visibility-filtered AP protocol).
        scores = [float(s) for s in det.scores]
        order = sorted(range(P), key=lambda i: -scores[i])
        for pi in order:
            ppoly = pred_polys[pi]
            if ppoly is None or ppoly.is_empty or not ppoly.is_valid:
                continue
            cx = float(det.corners[pi][:4, 0].mean())
            cy = float(det.corners[pi][:4, 1].mean())
            if not (ap_xmn <= cx <= ap_xmx and ap_ymn <= cy <= ap_ymx):
                pred_status[pi] = "OOR"
                continue
            best_iou = 0.0
            best_gi = -1
            for gi, gpoly in enumerate(gt_polys):
                if not gt_visible[gi]:
                    continue
                if gt_matched[gi] or gpoly is None or gpoly.is_empty or not gpoly.is_valid:
                    continue
                try:
                    inter = ppoly.intersection(gpoly).area
                    union = ppoly.union(gpoly).area
                except Exception:
                    continue
                if union <= 0:
                    continue
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            pred_iou[pi] = float(best_iou)
            if best_iou >= iou_thresh and best_gi >= 0:
                pred_status[pi] = "TP"
                gt_matched[best_gi] = True

        return gt_corners_ego, pred_status, pred_iou, gt_matched, gt_visible, gt_hits

    @classmethod
    def _project_ego_xy_to_top_img(cls, x_ego: float, y_ego: float):
        """Map an ego-frame ground point (x = forward, y = right) to pixel
        (u, v) in the top-down ``rgb_top`` camera image.

        The camera is mounted at ego (x=y=0), z=_TOP_Z, pitch=-90° so it looks
        straight down; with yaw=0 the camera rotates with the ego, so its
        image axes are: image-up = ego forward (+x), image-right = ego right
        (+y).

        Derivation:
            For pitch=-90 yaw=0, camera-frame coords for an ego-frame point
            (xe, ye, ze) are
                cam_x (right)   =  ye
                cam_y (down)    = -xe
                cam_z (forward) =  z_top - ze    # ground (ze≈0) → cam_z = z_top
            Pinhole:
                u = fx * cam_x / cam_z + W/2
                v = fy * cam_y / cam_z + H/2
            With fx = fy = W / (2 tan(FOV/2)), this collapses to
                u = (W/2) + (ye / z_top) * fx
                v = (H/2) - (xe / z_top) * fy

        Returns ``(u, v)`` in float pixels.
        """
        z_top = float(cls._TOP_Z)
        fx = cls._TOP_W / (2.0 * np.tan(np.radians(cls._TOP_FOV_DEG) / 2.0))
        fy = fx
        u = cls._TOP_W / 2.0 + (y_ego / z_top) * fx
        v = cls._TOP_H / 2.0 - (x_ego / z_top) * fy
        return float(u), float(v)

    def _save_bev_frame(self, primary_idx, ego_state, det, traj, cam_img=None,
                        ego_pose_world=None, top_img=None):
        """Side-by-side: front camera (left) + top-down BEV (right).

        BEV shows ego + route + 3-s plan + GT actors + predictions.
        Camera shows the live driver-view RGB if a camera sensor is declared.

        When PERCEPTION_SWAP_BEV_AP_ANNOT=1 (default), each predicted box is
        colored TP-green / FP-red at IoU >= PERCEPTION_SWAP_BEV_AP_IOU
        (default 0.5), each GT box is filled green if matched / gray if missed,
        and labels include the per-box detection score and best IoU.

        If ``top_img`` is the ``rgb_top`` CARLA frame (top-down RGB), the BEV
        pane uses that real image as its background instead of a synthetic
        matplotlib plot. When the sensor isn't delivered (e.g. on rigs that
        ran before this sensor was added), we fall back to the synthetic
        plot.
        """
        if not self._viz_dir:
            return
        if self._tick_count % self._viz_every != 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
        except Exception:
            return
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world = CarlaDataProvider.get_world()
            if world is None:
                return

            # Hoist last_step_info early so BOTH BEV variants (cam+top and
            # legacy world-coord) can reference it. The cam+top branch was
            # silently failing with "local variable 'info' referenced before
            # assignment" because info was only set 130 lines later in the
            # legacy branch — no _bev.png produced for those runs.
            info = self.last_step_info.get(primary_idx, {})

            ex, ey = float(ego_state.xy[0]), float(ego_state.xy[1])
            yaw = float(ego_state.yaw)
            cy, sy = np.cos(yaw), np.sin(yaw)
            # AP-eval range overlay (ego-frame XY). Drawn in both BEV variants.
            # Uses _ap_eval_range — the cropped scoring region (140×80 m by
            # default in openloop), not the detector's full cav_lidar_range
            # (204.8×204.8 m). The BEV must show what AP actually scores
            # against. Layout: [xmin, ymin, zmin, xmax, ymax, zmax].
            lr = self._ap_eval_range
            ap_range_xy = (float(lr[0]), float(lr[1]), float(lr[3]), float(lr[4]))

            # GT future ego trajectory matched to the planner's prediction
            # horizon (3 s @ 0.1 s = 30 waypoints). Sample at the same future
            # timestamps the planner rolls out (sim_t + (k+1)*dt for k=0..29).
            # Used both for the green-dashed BEV overlay and per-tick ADE.
            sim_t = float(getattr(self, "_last_sim_t", 0.0))
            gt_future_xy = None
            ade_per_tick = float("nan")
            fde_per_tick = float("nan")
            replay = self._gt_replays[primary_idx] if primary_idx < len(self._gt_replays) else None
            if replay is not None and traj is not None and len(traj) > 0:
                _gt_t_arr, _gt_xy_arr = replay
                _T = len(traj)
                _dt = 3.0 / max(_T, 1)  # mirrors to_trajectory(horizon_s=3.0, dt=0.1)
                # Sample GT at sim_t+(k+1)·dt for k=0..T-1, then anchor-shift
                # GT so its first sample lands on pred[0]. This nullifies
                # the start-offset (which is just a discretization artifact
                # in open-loop — pred and GT are both sampled +0.1 s into
                # the future from the same ego pose, so any offset there
                # carries no real signal). All other error sources — shape
                # divergence, speed-profile mismatch — are preserved.
                #   gt_shifted[k] = gt[k] - gt[0] + pred[0]
                #   err[k]        = ||pred[k] - gt_shifted[k]||
                #                 = ||(pred[k]-pred[0]) - (gt[k]-gt[0])||
                # i.e. relative-displacement L2 from step 0 onward.
                # NOTE: this differs from RiskM/opencood/utils/eval_utils.py
                # which uses raw L2 (no anchor shift). RiskM lives in an
                # ego-relative frame where start-offset is meaningful;
                # here we work in world frame with a sampling-induced
                # offset that has no physical meaning, so we correct for it.
                _samples = []
                _aligned_idx = []
                for _k in range(_T):
                    _xy = _gt_xy_at(_gt_t_arr, _gt_xy_arr,
                                    sim_t + (_k + 1) * _dt)
                    if _xy is not None:
                        _samples.append(_xy)
                        _aligned_idx.append(_k)
                if _samples:
                    gt_arr = np.asarray(_samples, dtype=np.float64)
                    pred_arr = np.asarray(
                        [traj[_k] for _k in _aligned_idx], dtype=np.float64,
                    )
                    gt_shifted = gt_arr - gt_arr[0] + pred_arr[0]
                    _diffs = np.linalg.norm(pred_arr - gt_shifted, axis=1)
                    ade_per_tick = float(_diffs.mean())
                    fde_per_tick = float(_diffs[-1])
                    # Render the SHIFTED GT so the bracket length on the
                    # BEV equals the labeled FDE numerically.
                    gt_future_xy = gt_shifted

            # Predictions: ego frame → world (rotation + translation by ego pose).
            pred_world = np.zeros((det.P, 2), dtype=np.float64) if det.P else None
            if det.P:
                for i, (px, py) in enumerate(det.centers_xy):
                    pred_world[i, 0] = ex + cy * px - sy * py
                    pred_world[i, 1] = ey + sy * px + cy * py

            # GT vehicles (excluding any hero/ego). Used both for legacy
            # scatter rendering and as a fallback when the AP matcher can't
            # produce ego-frame corners.
            ego_ids = self._ego_actor_ids or set()
            gt_xy = []
            for a in world.get_actors().filter("vehicle.*"):
                if a.id in ego_ids:
                    continue
                loc = a.get_location()
                gt_xy.append((loc.x, loc.y, a.attributes.get("role_name", "")))
            # Other heroes (cooperators) — show separately so we can see them.
            other_heroes = []
            for a in self._get_hero_actors():
                if a.id == self._get_hero_actors()[primary_idx].id:
                    continue
                loc = a.get_location()
                other_heroes.append((loc.x, loc.y))

            route_xy = self.last_step_info.get(primary_idx, {}).get("route_xy")

            # ----- Per-box AP match (TP/FP color, IoU + score) -----
            # `match` is None when annotation is disabled, in GT-oracle mode,
            # or when shapely / GT extraction fails. In any of those cases we
            # fall back to the original scatter rendering below.
            match = None
            if self._bev_ap_annot:
                match = self._compute_per_box_ap_match(
                    primary_idx, det, ego_pose_world,
                    iou_thresh=self._bev_ap_iou_thresh,
                )

            # Helper: rotate the (8,3) ego-frame corners' bottom face into
            # world coordinates for matplotlib polygon rendering.
            def _bottom_face_world(corners_ego):
                # corners_ego is already CCW in xy after _build_gt_corners_ego_frame
                # / boxes_to_corners_3d. Take the first 4 (bottom face).
                pts = []
                for k in range(4):
                    cx_ego = float(corners_ego[k][0])
                    cy_ego_ = float(corners_ego[k][1])
                    pts.append((
                        ex + cy * cx_ego - sy * cy_ego_,
                        ey + sy * cx_ego + cy * cy_ego_,
                    ))
                return pts

            # Two-panel figure: camera (left) + BEV (right).
            fig, (ax_cam, ax) = plt.subplots(1, 2, figsize=(16, 8),
                                              gridspec_kw={"width_ratios": [1, 1]})
            if cam_img is not None and hasattr(cam_img, "shape") and cam_img.size > 0:
                # CARLA delivers BGRA uint8; convert to RGB for matplotlib.
                if cam_img.ndim == 3 and cam_img.shape[-1] == 4:
                    cam_rgb = cam_img[..., [2, 1, 0]]
                else:
                    cam_rgb = cam_img
                ax_cam.imshow(cam_rgb)
            ax_cam.set_title(f"front cam · ego_{primary_idx}")
            ax_cam.set_xticks([])
            ax_cam.set_yticks([])

            # ----- BEV rendering -----
            # Two paths:
            #   (A) `top_img` available → render the actual CARLA top-down
            #       camera frame as the BEV background; project all geometry
            #       (route, plan, boxes, ego) into that image's pixel space.
            #   (B) `top_img` missing → fall back to the synthetic matplotlib
            #       BEV in world coordinates (legacy behavior).
            from matplotlib.patches import Polygon as MplPolygon
            from matplotlib.lines import Line2D

            use_topdown_bg = (
                top_img is not None and hasattr(top_img, "shape") and top_img.size > 0
            )
            n_tp = n_fp = n_fn = 0
            legend_handles = None

            if use_topdown_bg:
                # ----- (A) CARLA top-down BEV -----
                if top_img.ndim == 3 and top_img.shape[-1] == 4:
                    top_rgb = top_img[..., [2, 1, 0]]
                else:
                    top_rgb = top_img
                # `extent` aligns axes to image pixel coords; origin='upper'
                # makes (0,0) the top-left like the underlying image.
                ax.imshow(top_rgb, origin="upper",
                          extent=(0, self._TOP_W, self._TOP_H, 0))
                ax.set_xlim(0, self._TOP_W)
                ax.set_ylim(self._TOP_H, 0)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])

                # World-frame XY → ego-frame (forward, right) → top-down image px.
                def _world_to_uv(xw, yw):
                    dx, dy = xw - ex, yw - ey
                    # Ego forward = (cy, sy) world; ego right = (-sy, cy) world.
                    xe = cy * dx + sy * dy
                    ye = -sy * dx + cy * dy
                    return self._project_ego_xy_to_top_img(xe, ye)

                # Ego-frame XY → top-down image px (for predictions / GT).
                def _ego_to_uv(xe, ye):
                    return self._project_ego_xy_to_top_img(xe, ye)

                # Route (gray dashed)
                if route_xy is not None and len(route_xy) > 1:
                    uvs = [_world_to_uv(p[0], p[1]) for p in route_xy]
                    ax.plot([u for u, v in uvs], [v for u, v in uvs],
                            color="gray", linestyle="--", linewidth=1, alpha=0.7)
                # Both pred and GT future are sampled at sim_t + (k+1)·dt,
                # so neither traj[0] nor gt[0] is literally the ego's current
                # position — they're each one timestep into the future. With
                # high speed disagreement (e.g. brake_now while real ego
                # cruises) that 0.1 s offset becomes a ~5–10 px gap that the
                # eye reads as "the lines don't share an anchor." Prepend the
                # ego's current position to both lines so they visibly
                # emanate from the same point — the blue ego marker.
                ego_uv = _world_to_uv(ex, ey)

                # 3-s plan (orange) — predicted ego trajectory.
                # zorder=8 so it sits above the ego marker (zorder=7);
                # otherwise stationary or near-zero trajectories disappear
                # under the blue ego patch.
                pred_uv_end = None
                if traj is not None and len(traj) > 0:
                    uvs = [ego_uv] + [_world_to_uv(p[0], p[1]) for p in traj]
                    ax.plot([u for u, v in uvs], [v for u, v in uvs],
                            color="orange", linewidth=2.4, label="pred (3 s)",
                            zorder=8)
                    u_end, v_end = uvs[-1]
                    ax.plot([u_end], [v_end], "o", color="orange",
                            markersize=7, alpha=0.95, zorder=9)
                    pred_uv_end = (u_end, v_end)
                # GT future ego trajectory (green dashed) — same horizon as pred.
                # Per-step markers at every 5th step so a stationary GT still
                # shows visibly (otherwise the line collapses to a point under
                # the ego marker, which is exactly what hides ADE info at t=0).
                gt_uv_end = None
                if gt_future_xy is not None and len(gt_future_xy) > 0:
                    uvs = [ego_uv] + [_world_to_uv(p[0], p[1]) for p in gt_future_xy]
                    ax.plot([u for u, v in uvs], [v for u, v in uvs],
                            color="#1a8b2a", linestyle="--", linewidth=2.2,
                            alpha=0.95, label="GT future", zorder=8)
                    # Step markers — small dots every 5 steps + last.
                    step_idx = list(range(0, len(uvs), 5))
                    if (len(uvs) - 1) not in step_idx:
                        step_idx.append(len(uvs) - 1)
                    ax.plot(
                        [uvs[i][0] for i in step_idx],
                        [uvs[i][1] for i in step_idx],
                        "o", color="#1a8b2a", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.6,
                        alpha=0.95, zorder=9,
                    )
                    u_end, v_end = uvs[-1]
                    ax.plot([u_end], [v_end], "*", color="#1a8b2a",
                            markersize=12, markeredgecolor="white",
                            markeredgewidth=0.8, alpha=0.98, zorder=10)
                    gt_uv_end = (u_end, v_end)
                # Shared-anchor dot at ego_uv so the eye sees both lines
                # emerging from one point even at near-zero zoom.
                if (traj is not None and len(traj) > 0
                        or (gt_future_xy is not None and len(gt_future_xy) > 0)):
                    ax.plot(
                        [ego_uv[0]], [ego_uv[1]], "o", color="white",
                        markeredgecolor="black", markersize=4.5,
                        markeredgewidth=0.8, zorder=11,
                    )
                # Endpoint bracket: dotted line between pred-endpoint and
                # GT-endpoint. This is FDE geometrically (the final
                # displacement), NOT ADE — ADE is the mean over all 30
                # steps and has no single line on the plot. We label it
                # FDE here and surface ADE in the title + stat panel.
                # Per-step error visualization: thin segments connecting
                # pred[k] to gt[k] at every 5th step, so the eye can see
                # WHERE along the horizon the error is concentrated (those
                # 30 segments are what ADE averages over).
                if (pred_uv_end is not None and gt_uv_end is not None
                        and fde_per_tick == fde_per_tick):
                    # Per-step error segments — sample every 5 steps to
                    # avoid clutter, plus the last step for FDE emphasis.
                    if (traj is not None and gt_future_xy is not None
                            and len(traj) > 0 and len(gt_future_xy) > 0):
                        n_match = min(len(traj), len(gt_future_xy))
                        seg_idx = list(range(0, n_match, 5))
                        if (n_match - 1) not in seg_idx:
                            seg_idx.append(n_match - 1)
                        for _k in seg_idx:
                            pu, pv = _world_to_uv(
                                float(traj[_k][0]), float(traj[_k][1])
                            )
                            gu, gv = _world_to_uv(
                                float(gt_future_xy[_k][0]),
                                float(gt_future_xy[_k][1]),
                            )
                            ax.plot([pu, gu], [pv, gv],
                                    color="#7d2bff", linewidth=0.8,
                                    alpha=0.55, zorder=7)
                    # Endpoint bracket — labeled as FDE.
                    ax.plot(
                        [pred_uv_end[0], gt_uv_end[0]],
                        [pred_uv_end[1], gt_uv_end[1]],
                        ":", color="#7d2bff", linewidth=1.6, alpha=0.95,
                        zorder=8,
                    )
                    midu = 0.5 * (pred_uv_end[0] + gt_uv_end[0])
                    midv = 0.5 * (pred_uv_end[1] + gt_uv_end[1])
                    ade_part = (
                        f"\nADE={ade_per_tick:.1f}m"
                        if ade_per_tick == ade_per_tick else ""
                    )
                    ax.text(
                        midu, midv, f"FDE={fde_per_tick:.1f}m{ade_part}",
                        fontsize=7, ha="center", va="center", color="#7d2bff",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="#7d2bff",
                                  alpha=0.9, linewidth=0.7),
                        zorder=11,
                    )

                # GT + Pred boxes from the matcher (TP/FP color coded).
                # GT styling has 3 buckets: visible+matched (green), visible+
                # missed (gray = FN), non-visible (blue, dashed — filtered out
                # by the OPV2V semantic-LiDAR visibility protocol). Only
                # visible+missed counts as FN (matches AP semantics).
                n_nonvis = 0
                if match is not None:
                    gt_corners_ego, pred_status, pred_iou, gt_matched, gt_visible, gt_hits = match
                    for gi in range(len(gt_corners_ego)):
                        pts = [
                            _ego_to_uv(float(gt_corners_ego[gi][k][0]),
                                       float(gt_corners_ego[gi][k][1]))
                            for k in range(4)
                        ]
                        matched_ = bool(gt_matched[gi])
                        is_vis = bool(gt_visible[gi]) if gi < len(gt_visible) else True
                        if not is_vis:
                            edge = "#5a8cdc"
                            face = "#5a8cdc1a"
                            linestyle = "--"
                            n_nonvis += 1
                        elif matched_:
                            edge = "#1a8b2a"
                            face = "#1a8b2a33"
                            linestyle = "-"
                        else:
                            edge = "#bcbcbc"
                            face = "#bcbcbc1f"
                            linestyle = "-"
                            n_fn += 1
                        ax.add_patch(MplPolygon(
                            pts, closed=True, edgecolor=edge, facecolor=face,
                            linewidth=1.2, linestyle=linestyle, zorder=3,
                        ))
                        # Tiny hit-count tag so the user can see WHY a box was
                        # kept or dropped by the visibility filter and tune
                        # PERCEPTION_SWAP_VISIBILITY_FILTER_HITS accordingly.
                        h_count = int(gt_hits[gi]) if gi < len(gt_hits) else 0
                        cx_g = sum(p[0] for p in pts) / 4
                        cy_g = sum(p[1] for p in pts) / 4
                        ax.text(
                            cx_g, cy_g, f"h{h_count}",
                            fontsize=5, ha="center", va="center", color=edge,
                            zorder=4, alpha=0.9,
                        )
                    # Pred boxes — skip ego self-projection (cooperative-perception
                    # phantom on top of host ego). Same radius as the AP / planner
                    # filters so all three views are consistent. OOR preds (outside
                    # the cropped AP eval region) draw in faint orange and don't
                    # count toward TP/FP — they're shown for context only.
                    _self_r2 = self._self_filter_radius_m ** 2
                    n_oor = 0
                    for pi in range(det.P):
                        cx_ego = float(det.centers_xy[pi][0])
                        cy_ego = float(det.centers_xy[pi][1])
                        if (cx_ego * cx_ego + cy_ego * cy_ego) <= _self_r2:
                            continue
                        pts = [
                            _ego_to_uv(float(det.corners[pi][k][0]),
                                       float(det.corners[pi][k][1]))
                            for k in range(4)
                        ]
                        status = pred_status[pi] if pi < len(pred_status) else "FP"
                        if status == "OOR":
                            color = "#ffa552"
                            ax.add_patch(MplPolygon(
                                pts, closed=True, edgecolor=color, facecolor="none",
                                linewidth=1.0, linestyle=":", alpha=0.7, zorder=4,
                            ))
                            n_oor += 1
                            continue
                        is_tp = status == "TP"
                        color = "#28e64a" if is_tp else "#ff3232"
                        ax.add_patch(MplPolygon(
                            pts, closed=True, edgecolor=color, facecolor="none",
                            linewidth=1.8, zorder=5,
                        ))
                        cx_ = sum(p[0] for p in pts) / 4
                        cy_ = sum(p[1] for p in pts) / 4
                        score = float(det.scores[pi]) if pi < len(det.scores) else 0.0
                        label = (
                            f"{score:.2f}\n{pred_iou[pi]:.2f}"
                            if is_tp else f"{score:.2f}"
                        )
                        ax.text(cx_, cy_, label, fontsize=6, ha="center", va="center",
                                color=color, zorder=6,
                                bbox=dict(boxstyle="round,pad=0.1",
                                          facecolor="white", edgecolor="none", alpha=0.6))
                        if is_tp:
                            n_tp += 1
                        else:
                            n_fp += 1
                    legend_handles = [
                        Line2D([0], [0], color="#28e64a", lw=2, label=f"TP ({n_tp})"),
                        Line2D([0], [0], color="#ff3232", lw=2, label=f"FP ({n_fp})"),
                        Line2D([0], [0], color="#bcbcbc", lw=2, label=f"missed GT ({n_fn})"),
                        Line2D([0], [0], color="#5a8cdc", lw=2, linestyle="--",
                               label=f"non-visible GT ({n_nonvis})"),
                        Line2D([0], [0], color="#ffa552", lw=2, linestyle=":",
                               label=f"OOR pred ({n_oor})"),
                        Line2D([0], [0], color="orange", lw=2.2, label="pred (3 s)"),
                        Line2D([0], [0], color="#1a8b2a", lw=2.2, linestyle="--",
                               label="GT future"),
                    ]

                # Cooperator(s)
                for (ox, oy) in other_heroes:
                    u, v = _world_to_uv(ox, oy)
                    ax.plot([u], [v], "^", color="purple", markersize=12, alpha=0.85)

                # Ego marker (yaw-rotated arrowhead). In ego frame the ego is
                # at (0,0) facing +x, so we project the local arrow shape
                # directly via _ego_to_uv (no extra rotation needed).
                ego_len = 4.5
                ego_wid = 2.0
                ego_local = [
                    ( ego_len * 0.6,  0.0),
                    (-ego_len * 0.4,  ego_wid),
                    (-ego_len * 0.2,  0.0),
                    (-ego_len * 0.4, -ego_wid),
                ]
                ego_pts = [_ego_to_uv(lx, ly) for (lx, ly) in ego_local]
                ax.add_patch(MplPolygon(
                    ego_pts, closed=True, facecolor="blue", edgecolor="black",
                    linewidth=1.0, alpha=0.95, zorder=7,
                ))

                # AP-evaluation range overlay (detector cav_lidar_range, ego-
                # centered). HEAL configs use 204.8×204.8 m for AttFuse — that
                # extends well past the 120×120 m visible BEV viewport, so we
                # detect the "box covers entire viewport" case and draw arrow
                # markers at each image edge instead of an invisible polygon.
                xmn, ymn, xmx, ymx = ap_range_xy
                ap_box_local = [(xmn, ymn), (xmx, ymn), (xmx, ymx), (xmn, ymx)]
                ap_box_uv = [_ego_to_uv(lx, ly) for (lx, ly) in ap_box_local]
                _u_min = min(u for u, v in ap_box_uv)
                _u_max = max(u for u, v in ap_box_uv)
                _v_min = min(v for u, v in ap_box_uv)
                _v_max = max(v for u, v in ap_box_uv)
                ap_overflows = (
                    _u_min < 0 and _u_max > self._TOP_W
                    and _v_min < 0 and _v_max > self._TOP_H
                )
                if ap_overflows:
                    # Draw inset cyan arrows at each edge to convey "AP range
                    # extends past this view."
                    for (uu, vv, du, dv) in [
                        (self._TOP_W // 2, 14, 0, -1),
                        (self._TOP_W // 2, self._TOP_H - 14, 0, 1),
                        (14, self._TOP_H // 2, -1, 0),
                        (self._TOP_W - 14, self._TOP_H // 2, 1, 0),
                    ]:
                        ax.annotate(
                            "", xy=(uu + du * 12, vv + dv * 12),
                            xytext=(uu, vv),
                            arrowprops=dict(arrowstyle="->", color="#00bcd4",
                                            lw=1.6, alpha=0.9),
                            zorder=8,
                        )
                else:
                    ax.add_patch(MplPolygon(
                        ap_box_uv, closed=True, facecolor="none", edgecolor="#00bcd4",
                        linewidth=1.6, linestyle="--", alpha=0.9, zorder=8,
                        label=f"AP range ({xmx-xmn:.0f}×{ymx-ymn:.0f} m)",
                    ))

                # ── Prominent AP / ADE / CR stat panel on the BEV itself ──
                # The title bar gets crowded and easy to miss; a top-left
                # overlay puts the headline numbers right where you're
                # already looking.
                _running_ap_top = self._running_ap_at_iou(
                    primary_idx, self._bev_ap_iou_thresh
                )
                _ade_text = (
                    f"{ade_per_tick:.1f} m"
                    if ade_per_tick == ade_per_tick else "—"
                )
                _ttc = float(info.get("min_ttc", float("inf")))
                _cr_flag_top = 1 if (_ttc < 2.5 and _ttc != float("inf")) else 0
                _denom_p_top = max(1, n_tp + n_fp)
                _denom_r_top = max(1, n_tp + n_fn)
                _ap_iou_int = int(round(self._bev_ap_iou_thresh * 100))
                # Visibility-filter diagnostic: did the semantic LiDAR fire
                # this tick, and if so how many actors did it mark visible?
                _vis_set = (
                    (self._visible_ids_per_slot or {}).get(primary_idx)
                    if hasattr(self, "_visible_ids_per_slot") else None
                )
                _gt_total = (n_tp + n_fn + n_nonvis)  # in-range GT in this frame
                if _vis_set is None:
                    _vis_text = "vis filter: OFF (no sem-lidar this tick)"
                else:
                    _vis_kept = _gt_total - n_nonvis
                    _vis_text = (
                        f"vis filter: ON  kept={_vis_kept}/{_gt_total}  "
                        f"|sem-hits|={len(_vis_set)}"
                    )
                stat_lines = [
                    (
                        f"AP@{_ap_iou_int} (run): "
                        f"{_running_ap_top:.3f}"
                        if _running_ap_top is not None
                        else f"AP@{_ap_iou_int} (run): n/a"
                    ),
                    f"this frame: TP={n_tp}  FP={n_fp}  FN={n_fn}  "
                    f"P={n_tp/_denom_p_top:.2f}  R={n_tp/_denom_r_top:.2f}",
                    f"ADE: {_ade_text}    CR: {_cr_flag_top}"
                    + (f" (ttc={_ttc:.1f}s)" if _cr_flag_top else ""),
                    f"AP range: {xmx-xmn:.0f}×{ymx-ymn:.0f} m"
                    + ("  (extends past BEV)" if ap_overflows else ""),
                    _vis_text,
                ]
                ax.text(
                    8, 8, "\n".join(stat_lines),
                    fontsize=10, ha="left", va="top", color="black",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="black", alpha=0.85, linewidth=0.8),
                    zorder=12,
                )
            else:
                # ----- (B) Legacy synthetic BEV (matplotlib world coords) -----
                if route_xy is not None and len(route_xy) > 1:
                    ax.plot(route_xy[:, 0], route_xy[:, 1], color="gray",
                            linestyle="--", linewidth=1, alpha=0.6, label="route")
                if traj is not None and len(traj) > 0:
                    ax.plot(traj[:, 0], traj[:, 1], color="orange",
                            linewidth=2.4, label=f"pred (n={len(traj)})")
                    ax.plot(traj[-1, 0], traj[-1, 1], "o", color="orange",
                            markersize=6, alpha=0.7)
                if gt_future_xy is not None and len(gt_future_xy) > 0:
                    ax.plot(gt_future_xy[:, 0], gt_future_xy[:, 1],
                            color="#1a8b2a", linestyle="--", linewidth=2.0,
                            alpha=0.9, label=f"GT future (n={len(gt_future_xy)})")
                    ax.plot(gt_future_xy[-1, 0], gt_future_xy[-1, 1], "*",
                            color="#1a8b2a", markersize=10, alpha=0.95)
                n_nonvis = 0
                if match is not None:
                    gt_corners_ego, pred_status, pred_iou, gt_matched, gt_visible, gt_hits = match
                    for gi in range(len(gt_corners_ego)):
                        pts = _bottom_face_world(gt_corners_ego[gi])
                        matched_ = bool(gt_matched[gi])
                        is_vis = bool(gt_visible[gi]) if gi < len(gt_visible) else True
                        if not is_vis:
                            color = "#5a8cdc"
                            face = "#5a8cdc1a"
                            linestyle = "--"
                            n_nonvis += 1
                        elif matched_:
                            color = "#1a8b2a"
                            face = color + "33"
                            linestyle = "-"
                        else:
                            color = "#7f7f7f"
                            face = color + "33"
                            linestyle = "-"
                            n_fn += 1
                        ax.add_patch(MplPolygon(
                            pts, closed=True, edgecolor=color, facecolor=face,
                            linewidth=1.4, linestyle=linestyle, zorder=2,
                        ))
                    _self_r2 = self._self_filter_radius_m ** 2
                    n_oor = 0
                    for pi in range(det.P):
                        cx_ego = float(det.centers_xy[pi][0])
                        cy_ego = float(det.centers_xy[pi][1])
                        if (cx_ego * cx_ego + cy_ego * cy_ego) <= _self_r2:
                            continue
                        pts = _bottom_face_world(det.corners[pi])
                        status = pred_status[pi] if pi < len(pred_status) else "FP"
                        if status == "OOR":
                            ax.add_patch(MplPolygon(
                                pts, closed=True, edgecolor="#ffa552",
                                facecolor="none", linewidth=1.0, linestyle=":",
                                alpha=0.7, zorder=3,
                            ))
                            n_oor += 1
                            continue
                        is_tp = status == "TP"
                        color = "#1f9c2c" if is_tp else "#d62828"
                        ax.add_patch(MplPolygon(
                            pts, closed=True, edgecolor=color, facecolor="none",
                            linewidth=1.8, zorder=4,
                        ))
                        cx_ = sum(p[0] for p in pts) / 4
                        cy_ = sum(p[1] for p in pts) / 4
                        score = float(det.scores[pi]) if pi < len(det.scores) else 0.0
                        label = (
                            f"{score:.2f}\n{pred_iou[pi]:.2f}"
                            if is_tp else f"{score:.2f}"
                        )
                        ax.text(cx_, cy_, label, fontsize=6, ha="center", va="center",
                                color=color, zorder=5,
                                bbox=dict(boxstyle="round,pad=0.1",
                                          facecolor="white", edgecolor="none", alpha=0.6))
                        if is_tp:
                            n_tp += 1
                        else:
                            n_fp += 1
                    legend_handles = [
                        Line2D([0], [0], color="#1f9c2c", lw=2, label=f"TP ({n_tp})"),
                        Line2D([0], [0], color="#d62828", lw=2, label=f"FP ({n_fp})"),
                        Line2D([0], [0], color="#7f7f7f", lw=2, label=f"missed GT ({n_fn})"),
                        Line2D([0], [0], color="#5a8cdc", lw=2, linestyle="--",
                               label=f"non-visible GT ({n_nonvis})"),
                        Line2D([0], [0], color="#ffa552", lw=2, linestyle=":",
                               label=f"OOR pred ({n_oor})"),
                        Line2D([0], [0], color="orange", lw=2.2, label="pred (3 s)"),
                        Line2D([0], [0], color="#1a8b2a", lw=2.2, linestyle="--",
                               label="GT future"),
                    ]
                else:
                    if gt_xy:
                        gx = [g[0] for g in gt_xy]
                        gy = [g[1] for g in gt_xy]
                        ax.scatter(gx, gy, s=42, c="green", marker="s",
                                   edgecolors="black", linewidths=0.5,
                                   label=f"GT ({len(gt_xy)})")
                    if pred_world is not None and len(pred_world) > 0:
                        ax.scatter(pred_world[:, 0], pred_world[:, 1], s=80, c="red",
                                   marker="+", linewidths=2.0,
                                   label=f"pred ({len(pred_world)})")

                for (ox, oy) in other_heroes:
                    ax.plot(ox, oy, "^", color="purple", markersize=14, alpha=0.7,
                            label="cooperator" if other_heroes.index((ox, oy)) == 0 else None)

                # Yaw-rotated ego arrowhead (legacy world-coord path).
                ego_len = 4.5
                ego_wid = 2.0
                ego_local = [
                    ( ego_len * 0.6,  0.0),
                    (-ego_len * 0.4,  ego_wid),
                    (-ego_len * 0.2,  0.0),
                    (-ego_len * 0.4, -ego_wid),
                ]
                ego_world = [
                    (ex + cy * lx - sy * ly, ey + sy * lx + cy * ly)
                    for (lx, ly) in ego_local
                ]
                ax.add_patch(MplPolygon(
                    ego_world, closed=True, facecolor="blue", edgecolor="black",
                    linewidth=1.0, alpha=0.95, zorder=6,
                ))

                # AP-evaluation range overlay (RiskM XY box, rotated by ego yaw).
                xmn, ymn, xmx, ymx = ap_range_xy
                ap_box_local = [(xmn, ymn), (xmx, ymn), (xmx, ymx), (xmn, ymx)]
                ap_box_world = [
                    (ex + cy * lx - sy * ly, ey + sy * lx + cy * ly)
                    for (lx, ly) in ap_box_local
                ]
                ax.add_patch(MplPolygon(
                    ap_box_world, closed=True, facecolor="none", edgecolor="#00bcd4",
                    linewidth=1.6, linestyle="--", alpha=0.9, zorder=7,
                    label=f"AP range ({xmx-xmn:.0f}×{ymx-ymn:.0f} m)",
                ))

                # Window centred on ego, sized to fit the AP range box plus a
                # small margin so the dashed cyan rectangle is fully visible.
                R = max(abs(ap_range_xy[0]), abs(ap_range_xy[1]),
                        abs(ap_range_xy[2]), abs(ap_range_xy[3])) + 10.0
                ax.set_xlim(ex - R, ex + R)
                ax.set_ylim(ey - R, ey + R)
                ax.set_aspect("equal")
            # Grid only makes sense over the synthetic plot. Skip for the
            # CARLA top-down image background — gridlines over real pixels
            # are visual noise.
            if not use_topdown_bg:
                ax.grid(alpha=0.3)
            # `info` was hoisted to the top of the function — duplicate removed.
            ap_summary = ""
            if match is not None:
                # The AP number itself is the running cumulative AP@iou
                # across every frame seen so far for this ego — that IS the
                # AP a single tick can contribute to. (Per-tick AP doesn't
                # exist; AP is a PR-curve integral over recall thresholds,
                # not a single-frame quantity.) The TP/FP/FN/P/R counts that
                # follow describe THIS frame only — they're the per-frame
                # detector behavior that adds into the running AP.
                _denom_p = max(1, n_tp + n_fp)
                _denom_r = max(1, n_tp + n_fn)
                _running_ap = self._running_ap_at_iou(
                    primary_idx, self._bev_ap_iou_thresh
                )
                _ap_iou_int = int(round(self._bev_ap_iou_thresh * 100))
                _ap_value_str = (
                    f"{_running_ap:.3f}"
                    if _running_ap is not None else "n/a"
                )
                ap_summary = (
                    f"  AP@{_ap_iou_int}={_ap_value_str}"
                    f"  [frame: TP={n_tp} FP={n_fp} FN={n_fn} "
                    f"P={n_tp/_denom_p:.2f} R={n_tp/_denom_r:.2f}]"
                )
            # ADE: per-tick mean L2 of pred vs GT future (NaN if no GT loaded
            # or sim_t is past the recorded horizon).
            ade_str = (
                f"  ADE={ade_per_tick:.2f}m"
                if ade_per_tick == ade_per_tick  # NaN-safe
                else "  ADE=—"
            )
            # CR proxy: rule planner's min_ttc < ttc_brake_s (2.5s) means a
            # detected obstacle is on a collision course in the prediction
            # horizon. ``Inf`` (no in-corridor detections) => CR=0.
            ttc = float(info.get("min_ttc", float("inf")))
            cr_flag = 1 if (ttc < 2.5 and ttc != float("inf")) else 0
            cr_str = f"  CR={cr_flag}" + (
                f" (ttc={ttc:.1f}s)" if cr_flag else ""
            )
            ax.set_title(
                f"tick {self._tick_count:04d} ego_{primary_idx}  "
                f"spd={ego_state.speed:.1f} m/s  "
                f"P={det.P}  GT={len(gt_xy)}  "
                f"brake={'Y' if info.get('brake_now') else 'N'}"
                f"{ade_str}{cr_str}{ap_summary}",
                fontsize=10,
            )
            if legend_handles is not None:
                ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
            else:
                ax.legend(loc="upper right", fontsize=8)

            out_path = os.path.join(
                self._viz_dir,
                f"tick_{self._tick_count:05d}_ego{primary_idx}.png",
            )
            fig.savefig(out_path, dpi=80, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            # Don't let viz crash the drive.
            self._viz_err_count = getattr(self, "_viz_err_count", 0) + 1
            if self._viz_err_count <= 3:
                print(f"[perception_swap] viz frame failed: {exc}")

    def _running_ap_at_iou(self, primary_idx: int, iou_thresh: float):
        """Cumulative AP@iou across all frames seen so far by ego ``primary_idx``.

        Returns None if there's no AP collector for this ego, no frames
        recorded yet, or the matching IoU bucket isn't in the collector's
        threshold list. Mirrors the math used in ``CarlaGTApCollector.save``
        but doesn't write any file — purely for surfacing into the BEV
        title so the user can watch AP evolve over the run.
        """
        if primary_idx >= len(self.ap_collectors):
            return None
        coll = self.ap_collectors[primary_idx]
        if coll is None or coll.n_frames == 0:
            return None
        # Pick the closest threshold the collector actually tracks (handles
        # the user-facing 0.5 vs collector-internal 0.50 alias case).
        thresholds = list(coll.iou_thresholds)
        match = min(thresholds, key=lambda t: abs(float(t) - float(iou_thresh)))
        if abs(float(match) - float(iou_thresh)) > 1e-3:
            return None
        try:
            _, _, calculate_ap, _, _ = _lazy_eval_imports()
            if (coll.result_stat[match]["gt"] == 0
                    or not coll.result_stat[match]["tp"]):
                return 0.0
            ap_value, _, _ = calculate_ap(coll.result_stat, match)
            return float(ap_value)
        except Exception:
            return None

    def _resolve_ego_actor_ids(self) -> set:
        """Find the CARLA actor IDs for the hero (ego) vehicles. Used to
        exclude them from GT during AP collection."""
        actors = self._get_hero_actors()
        return {a.id for a in actors}

    def _get_hero_actors(self) -> List[Any]:
        """Hero vehicles in CARLA, ordered by their numeric role_name suffix.

        Leaderboard convention: ego vehicles have role_name 'hero', 'hero_1',
        'hero_2', ... corresponding to ego slots 0, 1, 2, ...
        """
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world = CarlaDataProvider.get_world()
            if world is None:
                return []
            heroes = [a for a in world.get_actors().filter("vehicle.*")
                      if a.attributes.get("role_name", "").startswith("hero")]
            def _slot(a):
                rn = a.attributes.get("role_name", "hero")
                if rn == "hero":
                    return 0
                if rn.startswith("hero_"):
                    try:
                        return int(rn.split("_", 1)[1])
                    except Exception:
                        return 0
                return 0
            heroes.sort(key=_slot)
            return heroes
        except Exception:
            return []

    def _gt_perception_pass(self, primary_pose_world, ego_actor_ids):
        """Replace one perception forward pass with a CARLA-ground-truth one.

        Returns a ``Detections``-shaped result (same fields as PerceptionRunner
        outputs) so all downstream code (Detection list, AP collector, BEV viz,
        diagnostics) keeps working unchanged.

        Range-gated to ``self._lidar_range`` so the oracle isn't unfairly
        omniscient — perception models only see things within sensor range,
        and GT must respect the same horizon for the comparison to be valid.
        """
        # Imports gated to avoid eager carla_data_provider load.
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from tools.perception_runner import Detections as _DetResult
        _, _, _, _, x_to_world = _lazy_eval_imports()

        world = CarlaDataProvider.get_world()
        empty = _DetResult(
            corners=np.zeros((0, 8, 3), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            centers_xy=np.zeros((0, 2), dtype=np.float32),
            P=0,
        )
        if world is None:
            return empty

        all_v = list(world.get_actors().filter("vehicle.*"))
        gt_actors = [a for a in all_v if a.id not in (ego_actor_ids or set())]
        if not gt_actors:
            return empty

        T_ego_world = np.linalg.inv(x_to_world(primary_pose_world))
        lr = self._lidar_range  # [xmin, ymin, zmin, xmax, ymax, zmax]

        corners_list: List[np.ndarray] = []
        for actor in gt_actors:
            try:
                world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            cw = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
            if cw.shape != (8, 3):
                continue
            hom = np.concatenate([cw, np.ones((8, 1))], axis=1)
            ce = (T_ego_world @ hom.T).T[:, :3]                        # (8, 3) ego frame
            # Reorder to clean CCW bottom face (same fix as the AP collector).
            order_z = np.argsort(ce[:, 2])
            bot, top = ce[order_z[:4]], ce[order_z[4:]]
            bc = bot[:, :2].mean(axis=0)
            ang = np.arctan2(bot[:, 1] - bc[1], bot[:, 0] - bc[0])
            bot = bot[np.argsort(ang)]
            mt = np.zeros_like(top)
            used = np.zeros(4, dtype=bool)
            for i in range(4):
                d = np.linalg.norm(top[:, :2] - bot[i, :2], axis=1)
                d[used] = np.inf
                j = int(np.argmin(d))
                used[j] = True
                mt[i] = top[j]
            cor = np.concatenate([bot, mt], axis=0)                    # (8, 3)

            c = cor.mean(axis=0)
            if not (lr[0] <= c[0] <= lr[3] and lr[1] <= c[1] <= lr[4]):
                continue
            corners_list.append(cor)

        if not corners_list:
            return empty
        corners = np.stack(corners_list, axis=0).astype(np.float32)
        centers_xy = corners.mean(axis=1)[:, :2].astype(np.float32)
        scores = np.ones((len(corners),), dtype=np.float32)
        return _DetResult(corners=corners, scores=scores,
                          centers_xy=centers_xy, P=len(corners))

    # ─────── Sensor declaration ───────────────────────────────────────
    # The harness reads this and feeds matching keys via input_data.
    # IDs are suffixed _<slot_id> by the harness; for ego 0 they become
    # `lidar_0`, `gps_0`, `imu_0`, `speed_0`.

    def sensors(self):
        # LiDAR config — yaw=0 so the sensor frame equals the ego frame and
        # the detector's predictions land directly in ego coords. Camera is
        # declared so we can save real driver-view images alongside the BEV
        # plot for visualization (only used by viz, not by the model).
        #
        # In open-loop (CUSTOM_EGO_LOG_REPLAY=1, set by --openloop) we swap
        # to OPV2V-paper-exact sensor params so AP comparison is apples-to-
        # apples with the trained-on distribution. Closed-loop runs keep the
        # legacy values so existing closed-loop behavior is unchanged.
        # OPV2V Table I (Xu et al., ICRA 2022): channels=64, range=120m,
        # upper_fov=+5, lower_fov=-25, points_per_second=1.3M. Mount-z is
        # not published in the paper; keep our z=2.5.
        if os.environ.get("CUSTOM_EGO_LOG_REPLAY", "0") == "1":
            lidar_kwargs = dict(
                range=120, rotation_frequency=20, channels=64,
                upper_fov=5, lower_fov=-25,
                points_per_second=1300000,
            )
        else:
            lidar_kwargs = dict(
                range=100, rotation_frequency=20, channels=64,
                upper_fov=4, lower_fov=-20,
                points_per_second=2304000,
            )
        sensors_list = [
            {"type": "sensor.lidar.ray_cast", "id": "lidar",
             "x": 0.0, "y": 0.0, "z": 2.5,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             **lidar_kwargs},
        ]
        # Semantic LiDAR for visibility-filtered AP (OPV2V protocol). Declared
        # whenever the visibility filter is enabled (default ON via
        # PERCEPTION_SWAP_VISIBILITY_FILTER_HITS=3). The previous gate also
        # required CUSTOM_EGO_LOG_REPLAY=1, but the new "closed-loop drive +
        # frame-0 metrics" pattern leaves that flag off, so the sensor was
        # silently absent and visibility was never computed. Same mount + ray
        # pattern as the main LiDAR; CARLA returns per-ray (x, y, z,
        # cos_inc_angle, object_idx, object_tag) and we count returns per actor
        # at GT-build time. Set PERCEPTION_SWAP_VISIBILITY_FILTER_HITS=0 to
        # opt out (e.g., for the strict-no-filter HEAL-Table-2 baseline).
        if int(os.environ.get("PERCEPTION_SWAP_VISIBILITY_FILTER_HITS", "3")) > 0:
            sensors_list.append({
                "type": "sensor.lidar.ray_cast_semantic", "id": "lidar_semantic",
                "x": 0.0, "y": 0.0, "z": 2.5,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                **lidar_kwargs,
            })
        sensors_list.extend([
            {"type": "sensor.camera.rgb", "id": "rgb_front",
             "x": 1.3, "y": 0.0, "z": 1.5,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 800, "height": 600, "fov": 100},
            # Top-down RGB camera (CARLA "BEV"). Mounted high above the ego
            # and pitched -90° so it looks straight down. Used as the BEV
            # background in the BEV viz pane (with TP/FP boxes overlaid).
            # Visible ground area = 2 * z * tan(fov/2) = 2 * 60 * tan(45°) = 120 m
            # Pixel resolution at z=60, fov=90, 800×800 = 0.15 m / px
            {"type": "sensor.camera.rgb", "id": "rgb_top",
             "x": 0.0, "y": 0.0, "z": 60.0,
             "roll": 0.0, "pitch": -90.0, "yaw": 0.0,
             "width": 800, "height": 800, "fov": 90.0},
            {"type": "sensor.other.gnss",     "id": "gps",
             "x": 0.0, "y": 0.0, "z": 0.0},
            {"type": "sensor.other.imu",      "id": "imu",
             "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"type": "sensor.speedometer",    "id": "speed",
             "reading_frequency": 20},
        ])
        return sensors_list

    # ─────── Global plan ──────────────────────────────────────────────
    # The harness calls this once before the first run_step. world_plan is a
    # list-of-lists (one list per ego) of (carla.Transform, RoadOption) pairs.

    def set_global_plan(self, gps_plan, world_plan):
        if not world_plan:
            raise RuntimeError("PerceptionSwapAgent: empty world_plan")
        if len(world_plan) != self.ego_vehicles_num:
            # Common pattern in this repo: world_plan is one route per ego.
            # If the harness gave us fewer routes than egos, broadcast the first.
            # Most cleanly: error out so we notice.
            raise RuntimeError(
                f"PerceptionSwapAgent: got {len(world_plan)} routes for "
                f"{self.ego_vehicles_num} egos"
            )

        for ego_i, ego_route in enumerate(world_plan):
            if not ego_route or len(ego_route) < 2:
                raise RuntimeError(
                    f"PerceptionSwapAgent: ego {ego_i} route too short "
                    f"({len(ego_route)} waypoints)"
                )
            route_xy = np.array([
                [float(transform.location.x), float(transform.location.y)]
                for transform, _cmd in ego_route
            ], dtype=np.float64)
            self.last_step_info.setdefault(ego_i, {})["route_xy"] = route_xy

            if self._planner_kind == "rule":
                self.planners[ego_i] = RulePlanner(
                    global_route=route_xy,
                    cruise_speed=self._cruise_speed,
                    brake_persistence_ticks=self._brake_persistence_ticks,
                    terminal_radius_m=self._terminal_radius_m,
                    tight_terminal_radius_m=self._tight_terminal_radius_m,
                    creep_speed_m_s=self._creep_speed_m_s,
                    creep_timeout_s=self._creep_timeout_s,
                    min_progress_m=self._terminal_min_progress_m,
                    lateral_clearance_m=self._lateral_clearance_m,
                    stopped_speed_m_s=self._stopped_speed_m_s,
                    stopped_min_gap_m=self._stopped_min_gap_m,
                )
            else:
                # BasicAgent needs the hero actor in its constructor; the
                # actor isn't spawned yet on the first set_global_plan call,
                # so we stash the route and build the planner lazily on the
                # first run_step that sees the actor.
                self._stored_world_plans[ego_i] = ego_route

    # ─────── Pose-noise injector (codriving-compatible) ───────────────

    def _apply_pose_noise(self, poses: List[List[float]], primary_idx: int) -> List[List[float]]:
        """Add Gaussian (or Laplace) noise to all agents' poses for fusion math.

        Mirrors `opencood.utils.pose_utils.add_noise_data_dict` exactly: each
        agent (primary + cooperators) gets an independent fresh noise sample
        on its [x, y, yaw_deg] components per tick. The relative pose error
        between primary and any cooperator is therefore the difference of
        two iid samples → std σ·√2, matching the relative noise distribution
        codriving's pipeline puts through its model.

        Why noise primary too: codriving noises every cav for the model's
        pairwise transforms but evaluates against GT using `lidar_pose_clean`
        (so primary's noise cancels in the metric and only the relative
        primary-cooperator error matters). The caller here keeps the clean
        primary pose around (`ego_poses_world[primary_idx]`) and uses it for
        the AP-collector world transform, so the same cancellation happens —
        only cooperator-feature warping is degraded, just like codriving.
        """
        if not self._pose_noise_active:
            return poses
        out = [list(p) for p in poses]
        for i in range(len(out)):
            if self._pose_noise_use_laplace:
                dx = float(np.random.laplace(self._pose_noise_pos_mean, self._pose_noise_pos_std))
                dy = float(np.random.laplace(self._pose_noise_pos_mean, self._pose_noise_pos_std))
                dyaw = float(np.random.laplace(self._pose_noise_rot_mean, self._pose_noise_rot_std))
            else:
                dx = float(np.random.normal(self._pose_noise_pos_mean, self._pose_noise_pos_std))
                dy = float(np.random.normal(self._pose_noise_pos_mean, self._pose_noise_pos_std))
                dyaw = float(np.random.normal(self._pose_noise_rot_mean, self._pose_noise_rot_std))
            out[i][0] = float(out[i][0]) + dx
            out[i][1] = float(out[i][1]) + dy
            out[i][4] = float(out[i][4]) + dyaw
        return out

    # ─────── Per-frame step ───────────────────────────────────────────

    def run_step(self, input_data: Dict[str, Any], timestamp: float):
        # Stash sim time for downstream viz / GT lookup. Leaderboard supplies
        # ``timestamp`` in seconds since scenario start — same time axis as
        # the REPLAY XML.
        self._last_sim_t = float(timestamp) if timestamp is not None else 0.0
        # Bail out cleanly if set_global_plan never landed. For the rule
        # planner we built it eagerly there; for basicagent we stashed the
        # route and will build the planner lazily once heroes are spawned.
        if self._planner_kind == "rule":
            if any(p is None for p in self.planners):
                return [_idle_control() for _ in range(self.ego_vehicles_num)]
        else:
            if any(p is None for p in self._stored_world_plans):
                return [_idle_control() for _ in range(self.ego_vehicles_num)]

        # 1. Read each ego's pose from CARLA directly (more reliable than
        # converting GNSS lat/lon → world meters), and read LiDAR from sensor.
        hero_actors = self._get_hero_actors()
        if len(hero_actors) < self.ego_vehicles_num:
            # Fall back to idle if we can't see the heroes yet — happens on
            # the very first tick before CARLA has registered them.
            return [_idle_control() for _ in range(self.ego_vehicles_num)]

        ego_states: List[EgoState] = []
        ego_lidars: List[np.ndarray] = []
        ego_poses_world: List[List[float]] = []

        for slot in range(self.ego_vehicles_num):
            actor = hero_actors[slot]
            tf = actor.get_transform()
            vel = actor.get_velocity()
            xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
            # CARLA Transform.rotation.yaw is in DEGREES (LH frame: +y is the
            # vehicle's RIGHT, so positive yaw rotates +x toward +y, which is
            # clockwise viewed from above). The numeric value (cos yaw, sin yaw)
            # gives the forward direction in CARLA world coords either way.
            yaw_deg = float(tf.rotation.yaw)
            yaw_rad = float(np.radians(yaw_deg))
            yaw_rad = float(((yaw_rad + np.pi) % (2.0 * np.pi)) - np.pi)
            speed = float(np.hypot(vel.x, vel.y))

            lidar = _read_sensor(input_data, "lidar", slot,
                                 default=np.zeros((0, 4), dtype=np.float32))

            # Read semantic LiDAR. CARLA's per-ray struct is
            #   x f32, y f32, z f32, cos_inc_angle f32, object_idx u32, object_tag u32
            # (24 bytes / 6 columns). The shared sensor parser reinterprets ALL
            # six columns as float32, which CORRUPTS object_idx — small actor
            # IDs (< 16M) become denormal floats < 1.0 that cast to int64 as 0,
            # making every ray look like it hit the static world. Recover the
            # actual integer IDs by viewing the bytes back as uint32.
            sem_lidar = _read_sensor(input_data, "lidar_semantic", slot,
                                     default=None)
            visible_ids: Optional[set] = None
            hits_per_actor: Dict[int, int] = {}
            if sem_lidar is not None and hasattr(sem_lidar, "shape") and sem_lidar.size > 0:
                try:
                    u32_view = np.ascontiguousarray(sem_lidar).view(np.uint32).reshape(-1, 6)
                    obj_ids_col = u32_view[:, 4].astype(np.int64)
                    # Drop hits on env (object_idx 0 = world). Count per actor.
                    obj_ids_col = obj_ids_col[obj_ids_col > 0]
                    if obj_ids_col.size > 0:
                        uniq, counts = np.unique(obj_ids_col, return_counts=True)
                        hits_per_actor = {int(a): int(c) for a, c in zip(uniq, counts)}
                        # Default 10: OPV2V protocol uses bev_visibility.png with
                        # similar density. min_hits=3 is too lenient — even
                        # heavily-occluded actors get 1-3 grazing rays through
                        # gaps between front-row vehicles. 10+ guarantees the
                        # actor's body is meaningfully painted by the LiDAR.
                        min_hits = int(os.environ.get(
                            "PERCEPTION_SWAP_VISIBILITY_FILTER_HITS", "10"))
                        visible_ids = {a for a, c in hits_per_actor.items() if c >= min_hits}
                except Exception as _e:
                    print(f"[perception_swap] sem-lidar parse failed (ego {slot}): {_e}")
            self._visible_ids_per_slot = getattr(self, "_visible_ids_per_slot", {})
            self._visible_ids_per_slot[slot] = visible_ids
            self._hits_per_actor_per_slot = getattr(self, "_hits_per_actor_per_slot", {})
            self._hits_per_actor_per_slot[slot] = hits_per_actor

            # One-shot sensor sanity print so we can verify the sensor config
            # produced the expected point count + intensity scale.
            if not getattr(self, f"_sensor_logged_{slot}", False) and len(lidar) > 0:
                print(f"[perception_swap] sensor sanity (ego {slot}, tick {self._tick_count}): "
                      f"lidar shape={lidar.shape}  "
                      f"x∈[{lidar[:,0].min():.1f},{lidar[:,0].max():.1f}] "
                      f"y∈[{lidar[:,1].min():.1f},{lidar[:,1].max():.1f}] "
                      f"z∈[{lidar[:,2].min():.1f},{lidar[:,2].max():.1f}] "
                      f"intensity∈[{lidar[:,3].min():.3f},{lidar[:,3].max():.3f}]")
                _sem_n = 0 if sem_lidar is None else int(getattr(sem_lidar, "size", 0))
                _vis_n = 0 if visible_ids is None else len(visible_ids)
                _sem_kind = (
                    "absent (sensor not declared or not delivered)"
                    if sem_lidar is None else f"{_sem_n // 6} rays  visible_actors={_vis_n}"
                )
                print(f"[perception_swap] sem-lidar (ego {slot}, tick {self._tick_count}): {_sem_kind}")
                if hits_per_actor:
                    # Hit-count histogram so we can see the distribution and
                    # pick a sane PERCEPTION_SWAP_VISIBILITY_FILTER_HITS.
                    _hcounts = sorted(hits_per_actor.values(), reverse=True)
                    _bins = [(1, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]
                    _hist = {f"{lo}-{hi if hi < 10**9 else '∞'}":
                             sum(1 for c in _hcounts if lo <= c <= hi) for lo, hi in _bins}
                    _top5 = ", ".join(f"id{a}:h{c}" for a, c in
                                      sorted(hits_per_actor.items(), key=lambda kv: -kv[1])[:5])
                    print(f"[perception_swap]   hit-count hist (ego {slot}): {_hist}")
                    print(f"[perception_swap]   top-5 actors by hits (ego {slot}): {_top5}")
                setattr(self, f"_sensor_logged_{slot}", True)

            ego_states.append(EgoState(xy=xy, yaw=yaw_rad, speed=speed))
            ego_lidars.append(np.asarray(lidar, dtype=np.float32))
            ego_poses_world.append([
                float(tf.location.x), float(tf.location.y), float(tf.location.z),
                float(tf.rotation.roll), yaw_deg, float(tf.rotation.pitch),
            ])

        # Append stationary infrastructure LiDAR streams (v2xpnp ablation).
        # No-op when --infra-collab is off or the map isn't ucla_v2. The
        # appended entries sit at indices [ego_vehicles_num, ...] so existing
        # primary_idx loops over [0, ego_vehicles_num) keep working. The
        # detect_cooperative path treats them identically to peer egos: same
        # voxelizer, same model batch, native intermediate-fusion forward.
        self._ensure_infra_sensors()
        _infra_lidars, _infra_poses = self._collect_infra_lidars()
        ego_lidars.extend(_infra_lidars)
        ego_poses_world.extend(_infra_poses)

        # ─── Communication latency: serve cooperator data from `latency_step`
        # ticks ago. Primary slots (each ego acts as primary in its own pass,
        # so any slot in [0, ego_vehicles_num) is "primary" for at least one
        # forward pass) keep fresh data — we substitute later, per-pass.
        # Here we just snapshot current data into the ring + materialize the
        # delayed payload, then patch the working lists wholesale; the
        # per-pass loop below restores fresh data for that pass's primary.
        # Mirrors colmdriver_action.py:1326-1343 latency convention.
        ego_lidars_fresh = list(ego_lidars)
        ego_poses_world_fresh = list(ego_poses_world)
        if self._comm_latency_steps > 0:
            self._latency_ring[self._tick_count] = (
                list(ego_lidars), list(ego_poses_world),
            )
            keep_after = self._tick_count - self._comm_latency_steps - 1
            for _k in [k for k in self._latency_ring if k < keep_after]:
                self._latency_ring.pop(_k, None)
            target_step = self._tick_count - self._comm_latency_steps
            delayed = self._latency_ring.get(target_step)
            if delayed is not None:
                d_lidars, d_poses = delayed
                # Positional substitution: index i in the current step
                # corresponds to the same agent at the delayed step (egos +
                # infra are stable across a scenario by construction). If the
                # cooperator set has shrunk/grown, only patch the overlap.
                n_overlap = min(len(d_lidars), len(ego_lidars))
                for i in range(n_overlap):
                    ego_lidars[i] = d_lidars[i]
                    ego_poses_world[i] = list(d_poses[i])

        # Lazy-build the BasicAgentPlanner now that heroes exist (one-shot).
        if self._planner_kind == "basicagent":
            for slot in range(self.ego_vehicles_num):
                if self.planners[slot] is None:
                    plan_for_slot = self._stored_world_plans[slot]
                    self.planners[slot] = BasicAgentPlanner(
                        actor=hero_actors[slot],
                        target_speed_kmh=self._target_speed_kmh,
                        brake_persistence_ticks=self._brake_persistence_ticks,
                        lateral_clearance_m=self._lateral_clearance_m,
                        stopped_speed_m_s=self._stopped_speed_m_s,
                    )
                    self.planners[slot].set_global_plan(plan_for_slot)
                    print(
                        f"[perception_swap] BasicAgentPlanner built for ego {slot} "
                        f"(target_speed={self._target_speed_kmh:.1f} km/h, "
                        f"route_len={len(plan_for_slot)})"
                    )
        elif self._planner_kind == "behavior":
            # OpenCDA BehaviorAgent (collision/overtake/car-follow rewired
            # to consume our Detection list). Same lazy-construction pattern
            # as BasicAgent: needs the hero actor in __init__.
            for slot in range(self.ego_vehicles_num):
                if self.planners[slot] is None:
                    plan_for_slot = self._stored_world_plans[slot]
                    self.planners[slot] = BehaviorAgentPlanner(
                        actor=hero_actors[slot],
                        target_speed_kmh=self._target_speed_kmh,
                        terminal_radius_m=self._terminal_radius_m,
                    )
                    self.planners[slot].set_global_plan(plan_for_slot)
                    print(
                        f"[perception_swap] BehaviorAgentPlanner built for ego {slot} "
                        f"(target_speed={self._target_speed_kmh:.1f} km/h, "
                        f"route_len={len(plan_for_slot)})"
                    )

        # GT-perception oracle path: skip the perception model entirely and
        # build Detections directly from world.get_actors(). One forward-pass-
        # equivalent per ego (so per-ego ego frames stay consistent).
        if self._use_gt_perception and self._ego_actor_ids is None:
            self._ego_actor_ids = self._resolve_ego_actor_ids()

        # 2. Run perception (or GT oracle) per ego, then plan.
        ctrls: List[carla.VehicleControl] = []
        for primary_idx in range(self.ego_vehicles_num):
            ego = ego_states[primary_idx]
            planner = self.planners[primary_idx]

            if self._use_gt_perception:
                det = self._gt_perception_pass(
                    ego_poses_world[primary_idx], self._ego_actor_ids
                )
            elif self._cooperative and len(ego_lidars) > 1:
                # len(ego_lidars) includes any appended infra-LiDAR streams,
                # so single-ego v2xpnp scenes still go through cooperative
                # fusion when --infra-collab is on.
                #
                # Latency: restore primary's slot to fresh data — primary
                # always knows its own state at t (no self-comm).
                # Pose noise: perturb non-primary [x, y, yaw] only.
                pass_lidars = list(ego_lidars)
                pass_poses  = [list(p) for p in ego_poses_world]
                if self._comm_latency_steps > 0:
                    if 0 <= primary_idx < len(pass_lidars):
                        pass_lidars[primary_idx] = ego_lidars_fresh[primary_idx]
                        pass_poses[primary_idx]  = list(ego_poses_world_fresh[primary_idx])
                if self._pose_noise_active:
                    pass_poses = self._apply_pose_noise(pass_poses, primary_idx)
                try:
                    det = self.perception.detect_cooperative(
                        lidars=pass_lidars,
                        agent_poses_world=pass_poses,
                        primary_idx=primary_idx,
                        comm_range_m=self._comm_range_m,
                        max_cooperators=self._max_cooperators,
                    )
                except IndexError as _exc:
                    # Heter perception model can hit "index 1 out of bounds
                    # for dimension 0 with size 1" on scenarios where the
                    # modality_feature_dict only has one ego's features (e.g.
                    # 2023-04-07-15-01-15_0_0 with disco): the model's
                    # heter_model_baseline.forward iterates expected egos but
                    # finds the dict short. Fall back to non-cooperative
                    # detection on the primary ego only — the run continues
                    # with single-ego perception instead of crashing.
                    print(f"[perception_swap_agent] cooperative detect IndexError: "
                          f"{_exc}; falling back to non-cooperative detect()")
                    det = self.perception.detect(ego_lidars[primary_idx])
            else:
                det = self.perception.detect(ego_lidars[primary_idx])

            # AP collection (CARLA GT). Skip in GT-oracle mode: pred == GT by
            # construction, so AP is trivially 1.0 and just costs cycles.
            # Lazy-init the ego ID set on first call so it picks up egos after
            # they're spawned.
            if (self._collect_ap and not self._use_gt_perception
                    and self.ap_collectors[primary_idx] is not None):
                if self._ego_actor_ids is None:
                    self._ego_actor_ids = self._resolve_ego_actor_ids()
                try:
                    # One-shot debug dump path: co-locate with the AP file by
                    # default so it survives across scenes (per-scene SAVE_PATH).
                    _debug_root = os.path.dirname(self._ap_out_path) if self._ap_out_path else "/tmp"
                    _debug_path = os.environ.get(
                        "PERCEPTION_SWAP_DEBUG_DUMP",
                        os.path.join(_debug_root, f"perception_swap_debug_dump_ego{primary_idx}.json"),
                    )
                    # Source-dataset frame index for the visible-actor
                    # filter. CARLA tick = 20 Hz, source dataset = 10 Hz
                    # (configurable via PERCEPTION_SWAP_REAL_DATASET_FPS).
                    # ``_last_sim_t`` is the elapsed sim time set in
                    # run_step; the source-dataset frame index is just
                    # round(sim_t * fps), with the collector clamping
                    # out-of-range indices to the nearest recorded frame.
                    _frame_idx = int(round(
                        float(getattr(self, "_last_sim_t", 0.0))
                        * self._real_dataset_fps
                    ))
                    _vis_actor_ids = (
                        (self._visible_ids_per_slot or {}).get(primary_idx)
                        if hasattr(self, "_visible_ids_per_slot") else None
                    )
                    # Hard-override: if OPENLOOP_DISABLE_VISIBILITY_FILTER is set,
                    # skip the live semantic-LiDAR filter too. Pairs with the
                    # _gt_from_real_dataset=False forced above so neither dataset
                    # nor sem-lidar visibility filters are applied.
                    if getattr(self, "_disable_visibility_filter", False):
                        _vis_actor_ids = None
                    self.ap_collectors[primary_idx].update(
                        primary_pose_world=ego_poses_world[primary_idx],
                        pred_corners_ego=det.corners,
                        pred_scores=det.scores,
                        lidar_range=self._ap_eval_range,
                        ego_actor_ids=self._ego_actor_ids,
                        debug_dump_path=_debug_path,
                        frame_idx=_frame_idx,
                        visible_actor_ids=_vis_actor_ids,
                    )
                except Exception as exc:
                    # Don't let AP-collection failures kill the drive. Rate-limit
                    # the printout so we don't drown the log.
                    self._ap_err_count = getattr(self, "_ap_err_count", 0) + 1
                    if self._ap_err_count <= 3 or self._ap_err_count % 100 == 0:
                        print(f"[perception_swap] AP update fail #{self._ap_err_count} "
                              f"(ego {primary_idx}): {exc}")

                # Snapshot AP summary every 10 ticks (~0.5 s @ 20 Hz / 1 s @ 10 Hz)
                # so the latest state survives even if the harness hard-exits before
                # destroy() completes — observed when scenario_runner's
                # remove_all_actors hits its 30 s budget on dense scenes (90+ walkers)
                # and bypasses the agent's destroy().
                if (self._tick_count + 1) % 10 == 0:
                    self._snapshot_ap_summary()

            # Drop ego self-detections (cooperative perception fuses the partner
            # ego's view of *this* ego into the host's output, producing a
            # vehicle-sized box at origin). Same radius the AP collector uses.
            _self_r2 = self._self_filter_radius_m ** 2
            det_list = [
                Detection(
                    xy_ego=det.centers_xy[i].astype(np.float64),
                    radius=self._detection_radius,
                    score=float(det.scores[i]),
                )
                for i in range(det.P)
                if (float(det.centers_xy[i][0]) ** 2 + float(det.centers_xy[i][1]) ** 2) > _self_r2
            ]

            # Optionally hide all detections from the planner so it never brakes.
            # AP collector still sees the real predictions below.
            planner_dets = [] if self._disable_brake else det_list

            # Plan + control. Two backends; plan_extras carries the path-specific
            # diagnostic fields so last_step_info / diag JSONL stay consistent.
            if self._planner_kind == "rule":
                intent = planner.plan(ego, planner_dets, dt=0.1)
                steer, throttle, brake = planner.to_control(intent, ego)
                traj = planner.to_trajectory(ego, intent, horizon_s=3.0, dt=0.1)
                plan_extras = {
                    "intent_speed": float(intent.target_speed),
                    "brake_now":    bool(intent.brake_now),
                    "min_ttc":      float(intent.debug.get("min_ttc", float("inf"))),
                    "cursor":       int(planner.cursor),
                }
            else:
                planner.set_detections(planner_dets)
                ctrl_carla = planner.run_step()
                steer    = float(ctrl_carla.steer)
                throttle = float(ctrl_carla.throttle)
                brake    = float(ctrl_carla.brake)
                traj     = planner.predict_trajectory(horizon_s=3.0, dt=0.1)
                plan_extras = {
                    "intent_speed": planner.target_speed_mps,
                    # BasicAgent's add_emergency_stop writes brake=max_brake (0.5).
                    # >= catches that exactly.
                    "brake_now":    brake >= 0.5,
                    "min_ttc":      float("inf"),
                    "cursor":       -1,
                }
            self.last_planned_waypoints_world = traj  # (T, 2) world-frame xy

            # Optional per-tick prediction-file output for the openloop prototype's
            # metric reader (off unless PERCEPTION_SWAP_WRITE_PRED_FILES=1).
            self._save_per_step_pred_files(
                primary_idx, ego, det, traj, ego_poses_world[primary_idx]
            )

            # Optional viz dump: front camera (raw) + combined BEV+cam every N ticks.
            if self._viz_dir:
                cam_img = _read_sensor(input_data, "rgb_front", primary_idx,
                                       default=None)
                top_img = _read_sensor(input_data, "rgb_top", primary_idx,
                                       default=None)
                # One-shot print so we know whether the camera sensors are alive.
                if not getattr(self, f"_cam_logged_{primary_idx}", False):
                    if cam_img is None or not hasattr(cam_img, "shape"):
                        print(f"[perception_swap] WARN: rgb_front_{primary_idx} not delivered by harness")
                    else:
                        print(f"[perception_swap] rgb_front_{primary_idx} shape={cam_img.shape} dtype={cam_img.dtype}")
                    if top_img is None or not hasattr(top_img, "shape"):
                        print(f"[perception_swap] WARN: rgb_top_{primary_idx} not delivered — BEV will fall back to matplotlib plot")
                    else:
                        print(f"[perception_swap] rgb_top_{primary_idx} shape={top_img.shape} dtype={top_img.dtype}")
                    setattr(self, f"_cam_logged_{primary_idx}", True)
                self._save_camera_frame(primary_idx, cam_img)
                self._save_camera_frame_annotated(
                    primary_idx, cam_img, det, ego_poses_world[primary_idx]
                )
                self._save_bev_frame(
                    primary_idx, ego, det, traj, cam_img,
                    top_img=top_img,
                    ego_pose_world=ego_poses_world[primary_idx],
                )

            self.last_step_info[primary_idx] = {
                "route_xy":      self.last_step_info.get(primary_idx, {}).get("route_xy"),
                "ego_xy":        ego.xy,
                "ego_yaw":       ego.yaw,
                "ego_speed":     ego.speed,
                "n_detections":  det.P,
                "intent_speed":  plan_extras["intent_speed"],
                "brake_now":     plan_extras["brake_now"],
                "min_ttc":       plan_extras["min_ttc"],
                "cursor":        plan_extras["cursor"],
                "cooperative":   self._cooperative and self.ego_vehicles_num > 1,
                "planner_kind":  self._planner_kind,
                "use_gt":        self._use_gt_perception,
                "traj_first":    traj[0].tolist(),
                "traj_last":     traj[-1].tolist(),
                "horizon_s":     3.0,
            }

            ctrl = carla.VehicleControl()
            ctrl.steer    = float(steer)
            ctrl.throttle = float(throttle)
            ctrl.brake    = float(brake)
            ctrl.hand_brake = False
            ctrl.manual_gear_shift = False
            ctrls.append(ctrl)

            # Per-frame diagnostic dump (one JSON line per ego per tick).
            if self._diag_fp is not None:
                try:
                    # Corridor stats only meaningful for the rule planner (it
                    # exposes look_ahead / corridor_half_width). For BasicAgent
                    # we use a reasonable proxy (BasicAgent.base_vehicle_threshold).
                    if self._planner_kind == "rule":
                        in_corridor_x_max = float(planner.look_ahead)
                        corridor_half_w   = float(planner.corridor_half_width)
                    else:
                        in_corridor_x_max = float(planner._agent._base_vehicle_threshold) + float(ego.speed)
                        corridor_half_w   = float(planner._corridor_half_width)
                    # Cooperator counts for the infra-collab ablation. infra
                    # entries sit at indices [ego_vehicles_num, ...] of
                    # ego_lidars (we extended that list earlier in run_step),
                    # so n_infra is just the difference.
                    _n_total_in_lidars = int(len(ego_lidars))
                    _n_infra = max(0, _n_total_in_lidars - int(self.ego_vehicles_num))
                    self._diag_fp.write(json.dumps({
                        "tick":         self._tick_count,
                        "ego":          int(primary_idx),
                        "planner_kind": self._planner_kind,
                        "use_gt":       self._use_gt_perception,
                        "ego_xy":       [float(ego.xy[0]), float(ego.xy[1])],
                        "ego_yaw":      float(ego.yaw),
                        "ego_speed":    float(ego.speed),
                        "n_dets":       int(det.P),
                        "n_dets_in_corridor": int(sum(
                            1 for d in det_list
                            if 0 <= d.xy_ego[0] <= in_corridor_x_max and
                               abs(d.xy_ego[1]) <= corridor_half_w + d.radius
                        )),
                        "intent_speed": float(plan_extras["intent_speed"]),
                        "brake_now":    bool(plan_extras["brake_now"]),
                        "min_ttc":      float(plan_extras["min_ttc"]),
                        "cursor":       int(plan_extras["cursor"]),
                        "ctrl":         {"steer": float(steer), "throttle": float(throttle), "brake": float(brake)},
                        # Infra-collab proof fields:
                        "n_egos":       int(self.ego_vehicles_num),
                        "n_infra_lidars": _n_infra,
                        "n_cooperators_in_fusion": _n_total_in_lidars,
                        "infra_collab_env":     bool(is_infra_collab_enabled()),
                        "cooperative_branch":   bool(
                            self._cooperative and _n_total_in_lidars > 1
                            and not self._use_gt_perception
                        ),
                    }) + "\n")
                    self._diag_fp.flush()
                except Exception:
                    pass

        self._tick_count += 1
        return ctrls


# ──────────────────────────────────────────────────────────────────────────
# Helpers (module-level so they're easy to unit-test)
# ──────────────────────────────────────────────────────────────────────────


def _load_conf(path: str) -> Dict[str, Any]:
    """Read a tiny YAML conf file. Empty/missing path → defaults."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PerceptionSwapAgent: missing conf file {path!r}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _read_sensor(input_data: Dict[str, Any], sensor_id: str, slot: int, default):
    """Return the data tensor of the (frame_idx, payload) tuple for `<sensor_id>_<slot>`."""
    key = f"{sensor_id}_{slot}"
    entry = input_data.get(key)
    if entry is None:
        return default
    return np.asarray(entry[1])


def _read_speed(input_data: Dict[str, Any], slot: int) -> float:
    key = f"speed_{slot}"
    entry = input_data.get(key)
    if entry is None:
        return 0.0
    payload = entry[1]
    if isinstance(payload, dict):
        return float(payload.get("speed", 0.0))
    try:
        return float(payload)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────
# CARLA-GT AP Collector
# ──────────────────────────────────────────────────────────────────────────
# In closed-loop CARLA we have perfect GT every tick: every spawned vehicle's
# pose and bounding box are queryable from CarlaDataProvider. We query each
# tick, project to the primary ego's LiDAR frame, and accumulate TP/FP using
# OpenCOOD's eval utilities. AP is computed at scenario end.

# Lazy-imported to avoid pulling torch/opencood when this file is loaded for
# argument parsing or doc generation.
def _lazy_eval_imports():
    import torch
    from opencood.utils.eval_utils import caluclate_tp_fp, calculate_ap
    from opencood.utils.box_utils import boxes_to_corners_3d
    from opencood.utils.transformation_utils import x_to_world
    return torch, caluclate_tp_fp, calculate_ap, boxes_to_corners_3d, x_to_world


class CarlaGTApCollector:
    """Per-detector AP accumulator using CARLA ground truth as oracle.

    Usage pattern inside the agent:
        collector = CarlaGTApCollector(iou_thresholds=(0.3, 0.5, 0.7))
        ...each tick, per primary ego...
            collector.update(
                primary_pose_world=[x, y, z, roll, yaw_deg, pitch],
                pred_corners_ego=det.corners,
                pred_scores=det.scores,
                lidar_range=runner.config[...]['cav_lidar_range'],
                ego_actor_ids={ego_id_0, ego_id_1, ...},  # to exclude egos from GT
            )
        ...on destroy()...
            collector.save(json_path)
    """
    def __init__(self, iou_thresholds=(0.1, 0.3, 0.5, 0.7),
                 self_filter_radius_m: float = 2.0,
                 visible_ids_per_frame: Optional[Dict[int, set]] = None):
        self.iou_thresholds = tuple(iou_thresholds)
        self.self_filter_radius_m = float(self_filter_radius_m)
        # When set, restrict GT to actors whose ``role_name`` attribute is in
        # ``visible_ids_per_frame[frame_idx]`` for the frame passed to update().
        # This is how we score AP only against the actors the real-world ego
        # actually saw in v2x-real-pnp, instead of every CARLA vehicle in the
        # simulator (which includes parked traffic + occluded actors the real
        # sensor never reported).
        self.visible_ids_per_frame: Optional[Dict[int, set]] = (
            visible_ids_per_frame
        )
        self.result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
                            for iou in self.iou_thresholds}
        self.n_frames = 0
        self.cum_n_pred = 0
        self.cum_n_pred_after_self_filter = 0
        self.cum_n_gt = 0
        self.cum_n_self_dets = 0
        # Diagnostic: how often we end up with 0 visible IDs because the
        # frame_idx fell outside the recorded range (helpful for debugging
        # tick→frame alignment issues).
        self.cum_frames_with_filter = 0
        self.cum_frames_filter_empty = 0

    def update(self,
               primary_pose_world,
               pred_corners_ego,
               pred_scores,
               lidar_range,
               ego_actor_ids,
               debug_dump_path: Optional[str] = None,
               frame_idx: Optional[int] = None,
               visible_actor_ids: Optional[set] = None):
        """Append one frame's TP/FP stats. If `debug_dump_path` is given AND we
        haven't dumped yet AND this frame has both preds and GTs, dump them.

        If ``frame_idx`` is given AND ``self.visible_ids_per_frame`` is set,
        GT is restricted to actors whose ``role_name`` is in the visible set
        for that frame (with nearest-frame fallback if frame_idx is missing).

        If ``visible_actor_ids`` is non-None it is treated as the OPV2V-style
        semantic-LiDAR visibility filter (set of CARLA actor IDs).
        """
        torch, caluclate_tp_fp, _, _, _ = _lazy_eval_imports()

        visible_ids = self._visible_ids_for_frame(frame_idx)
        gt_corners_ego = self._build_gt_corners_ego_frame(
            primary_pose_world=primary_pose_world,
            lidar_range=lidar_range,
            ego_actor_ids=ego_actor_ids,
            visible_ids=visible_ids,
            visible_actor_ids=visible_actor_ids,
        )

        # ── One-shot debug dump: first frame with both preds AND GT ──
        if (debug_dump_path is not None
                and not getattr(self, "_debug_dumped", False)
                and len(pred_corners_ego) > 0 and len(gt_corners_ego) > 0):
            try:
                pred_centers = pred_corners_ego.mean(axis=1)
                gt_centers   = gt_corners_ego.mean(axis=1)
                # For EACH pred (sorted by score desc), the closest GT.
                order = np.argsort(-pred_scores)
                preds_dump = []
                for rank, i in enumerate(order):
                    d = np.linalg.norm(gt_centers[:, :2] - pred_centers[i, :2], axis=1)
                    j = int(np.argmin(d))
                    preds_dump.append({
                        "rank":            rank,
                        "pred_xy":         pred_centers[i, :2].tolist(),
                        "pred_score":      float(pred_scores[i]),
                        "pred_dim_xy":     [
                            float(pred_corners_ego[i, :, 0].max() - pred_corners_ego[i, :, 0].min()),
                            float(pred_corners_ego[i, :, 1].max() - pred_corners_ego[i, :, 1].min()),
                        ],
                        "nearest_gt_xy":   gt_centers[j, :2].tolist(),
                        "nearest_gt_dim":  [
                            float(gt_corners_ego[j, :, 0].max() - gt_corners_ego[j, :, 0].min()),
                            float(gt_corners_ego[j, :, 1].max() - gt_corners_ego[j, :, 1].min()),
                        ],
                        "dist_m":          float(d[j]),
                    })
                # All GT centers + dims for full picture.
                gt_dump = []
                for k in range(len(gt_centers)):
                    gt_dump.append({
                        "gt_xy":  gt_centers[k, :2].tolist(),
                        "gt_dim": [
                            float(gt_corners_ego[k, :, 0].max() - gt_corners_ego[k, :, 0].min()),
                            float(gt_corners_ego[k, :, 1].max() - gt_corners_ego[k, :, 1].min()),
                        ],
                    })
                with open(debug_dump_path, "w") as f:
                    json.dump({
                        "n_pred":     int(len(pred_corners_ego)),
                        "n_gt":       int(len(gt_corners_ego)),
                        "primary_pose_world": list(primary_pose_world),
                        "lidar_range": list(lidar_range),
                        "all_preds_sorted_by_score": preds_dump,
                        "all_gt": gt_dump,
                    }, f, indent=2)
                print(f"[perception_swap] one-shot debug dump → {debug_dump_path}")
                self._debug_dumped = True
            except Exception as exc:
                print(f"[perception_swap] debug dump failed: {exc}")

        # Filter pred to within lidar range to match what GT does, and drop
        # ego self-detections (predictions whose center is near origin —
        # the model often emits a high-confidence box for the ego itself
        # because the ego's LiDAR sees its own car body returns).
        n_self_filtered = 0
        if len(pred_corners_ego) > 0:
            centers = pred_corners_ego.mean(axis=1)
            in_range = (
                (centers[:, 0] >= lidar_range[0]) & (centers[:, 0] <= lidar_range[3]) &
                (centers[:, 1] >= lidar_range[1]) & (centers[:, 1] <= lidar_range[4])
            )
            r2 = self.self_filter_radius_m ** 2
            not_self = (centers[:, 0] ** 2 + centers[:, 1] ** 2) > r2
            n_self_filtered = int((~not_self).sum())
            keep = in_range & not_self
            pred_corners_ego = pred_corners_ego[keep]
            pred_scores      = pred_scores[keep]
        self.cum_n_self_dets += n_self_filtered
        self.cum_n_pred_after_self_filter += int(len(pred_corners_ego))

        pc = torch.from_numpy(pred_corners_ego) if len(pred_corners_ego) else None
        ps = torch.from_numpy(pred_scores)      if len(pred_scores)      else None
        gc = torch.from_numpy(gt_corners_ego)   # (M, 8, 3)

        for iou in self.iou_thresholds:
            caluclate_tp_fp(pc, ps, gc, self.result_stat, iou_thresh=iou)

        self.n_frames   += 1
        self.cum_n_pred += int(len(pred_corners_ego))
        self.cum_n_gt   += int(len(gt_corners_ego))

    def _visible_ids_for_frame(self, frame_idx: Optional[int]):
        """Return the set of role_names visible at ``frame_idx`` per the
        v2x-real-pnp source dataset, or ``None`` to disable filtering.

        Falls back to the nearest recorded frame if the requested frame_idx
        is outside the recorded range — this absorbs minor tick→frame
        rounding without silently dropping all GT.
        """
        if self.visible_ids_per_frame is None or frame_idx is None:
            return None
        if frame_idx in self.visible_ids_per_frame:
            return self.visible_ids_per_frame[frame_idx]
        keys = self.visible_ids_per_frame.keys()
        if not keys:
            return set()
        nearest = min(keys, key=lambda k: abs(k - frame_idx))
        return self.visible_ids_per_frame[nearest]

    def _build_gt_world_with_meta(self, ego_actor_ids, visible_actor_ids=None):
        """Snapshot every CARLA vehicle (excluding egos) in WORLD frame plus
        per-actor metadata, for per-tick GT cache dumps. No range filter — the
        downstream AP recompute applies its own range. Returns
        (corners_world (M, 8, 3) float32,
         actor_ids   (M,) int64,
         role_names  (M,) str,
         type_ids    (M,) str)
        with corners ordered bottom-CCW first / top second (same convention as
        ``_build_gt_corners_ego_frame``).

        If ``visible_actor_ids`` is non-None it must be a set of actor IDs that
        the ego's semantic LiDAR hit at least N times this tick (OPV2V-style
        visibility filter); only those actors are kept in the GT set. If None,
        no visibility filter is applied (HEAL-strict protocol)."""
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        world = CarlaDataProvider.get_world()
        if world is None:
            return (np.zeros((0, 8, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64), [], [])
        all_vehicles = list(world.get_actors().filter("vehicle.*"))
        gt_actors = [a for a in all_vehicles if a.id not in ego_actor_ids]
        if visible_actor_ids is not None:
            gt_actors = [a for a in gt_actors if a.id in visible_actor_ids]
        if not gt_actors:
            return (np.zeros((0, 8, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64), [], [])
        out_corners, out_ids, out_roles, out_types = [], [], [], []
        for actor in gt_actors:
            try:
                world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            corners_w = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
            if corners_w.shape != (8, 3):
                continue
            # Order bottom-CCW first, top second (same as _build_gt_corners_ego_frame).
            order_z = np.argsort(corners_w[:, 2])
            bot = corners_w[order_z[:4]]
            top = corners_w[order_z[4:]]
            bot_c = bot[:, :2].mean(axis=0)
            ang = np.arctan2(bot[:, 1] - bot_c[1], bot[:, 0] - bot_c[0])
            bot = bot[np.argsort(ang)]
            matched_top = np.zeros_like(top)
            used = np.zeros(4, dtype=bool)
            for i in range(4):
                d = np.linalg.norm(top[:, :2] - bot[i, :2], axis=1)
                d[used] = np.inf
                j = int(np.argmin(d))
                used[j] = True
                matched_top[i] = top[j]
            out_corners.append(np.concatenate([bot, matched_top], axis=0).astype(np.float32))
            out_ids.append(int(actor.id))
            out_roles.append(str(actor.attributes.get("role_name", "")))
            out_types.append(str(actor.type_id))
        if not out_corners:
            return (np.zeros((0, 8, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64), [], [])
        return (np.stack(out_corners, axis=0),
                np.array(out_ids, dtype=np.int64),
                out_roles, out_types)

    def _build_gt_corners_ego_frame(self, primary_pose_world, lidar_range,
                                    ego_actor_ids, visible_ids=None,
                                    visible_actor_ids=None,
                                    return_actor_ids=False):
        """Query the CARLA world for vehicles, project to primary ego LiDAR frame.

        If ``visible_ids`` is a non-None set, GT is restricted to actors whose
        ``role_name`` attribute is in that set — used to mirror the real-world
        ego's actually-observed actor subset from v2x-real-pnp YAMLs.

        If ``visible_actor_ids`` is a non-None set of CARLA actor IDs (int),
        GT is additionally restricted to those IDs — used by the semantic-
        LiDAR visibility filter (OPV2V protocol).

        If ``return_actor_ids`` is True, returns ``(corners, actor_ids)``
        where ``actor_ids`` is an int64 ndarray aligned with ``corners``.
        """
        # Imports gated to avoid eager carla_data_provider load.
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        _, _, _, _, x_to_world = _lazy_eval_imports()

        world = CarlaDataProvider.get_world()
        if world is None:
            empty = np.zeros((0, 8, 3), dtype=np.float32)
            return (empty, np.zeros((0,), dtype=np.int64)) if return_actor_ids else empty

        all_vehicles = list(world.get_actors().filter("vehicle.*"))
        # Exclude egos from GT (we don't want to "detect" ourselves).
        gt_actors = [a for a in all_vehicles if a.id not in ego_actor_ids]
        if visible_ids is not None:
            self.cum_frames_with_filter += 1
            if not visible_ids:
                self.cum_frames_filter_empty += 1
            gt_actors = [
                a for a in gt_actors
                if str(a.attributes.get("role_name", "")) in visible_ids
            ]
        if visible_actor_ids is not None:
            gt_actors = [a for a in gt_actors if a.id in visible_actor_ids]
        if not gt_actors:
            empty = np.zeros((0, 8, 3), dtype=np.float32)
            return (empty, np.zeros((0,), dtype=np.int64)) if return_actor_ids else empty

        # Build world→ego_lidar transform.
        T_world_ego = x_to_world(primary_pose_world)        # ego→world
        T_ego_world = np.linalg.inv(T_world_ego)            # world→ego

        # CARLA's get_world_vertices() does NOT return a clean (-,-)→(+,-)→
        # (+,+)→(-,+) bottom face. The first 4 vertices alternate between
        # bottom and top of one x-face. The simple constant permutation we
        # tried earlier did not produce a valid CCW quadrilateral. Robust fix:
        #   1) project all 8 corners into ego frame,
        #   2) take the 4 with smallest z = bottom face,
        #   3) sort them CCW around their centroid in xy,
        #   4) match top corners to the same CCW order.
        # This guarantees corners[:4] is a clean polygon for shapely's IoU.

        all_corners_ego: List[np.ndarray] = []
        all_actor_ids: List[int] = []
        for actor in gt_actors:
            try:
                world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            corners_w = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
            if corners_w.shape != (8, 3):
                continue

            ones = np.ones((8, 1), dtype=np.float64)
            hom_w = np.concatenate([corners_w, ones], axis=1)
            hom_e = (T_ego_world @ hom_w.T).T
            pts = hom_e[:, :3]                                           # (8, 3)

            order_z = np.argsort(pts[:, 2])
            bot = pts[order_z[:4]]
            top = pts[order_z[4:]]
            bot_c = bot[:, :2].mean(axis=0)
            ang = np.arctan2(bot[:, 1] - bot_c[1], bot[:, 0] - bot_c[0])
            bot = bot[np.argsort(ang)]
            matched_top = np.zeros_like(top)
            used = np.zeros(4, dtype=bool)
            for i in range(4):
                d = np.linalg.norm(top[:, :2] - bot[i, :2], axis=1)
                d[used] = np.inf
                j = int(np.argmin(d))
                used[j] = True
                matched_top[i] = top[j]
            corn_ego = np.concatenate([bot, matched_top], axis=0)

            # Filter to lidar range based on centroid.
            c = corn_ego.mean(axis=0)
            if not (lidar_range[0] <= c[0] <= lidar_range[3] and
                    lidar_range[1] <= c[1] <= lidar_range[4]):
                continue
            all_corners_ego.append(corn_ego.astype(np.float32))
            all_actor_ids.append(int(actor.id))

        if not all_corners_ego:
            empty = np.zeros((0, 8, 3), dtype=np.float32)
            return (empty, np.zeros((0,), dtype=np.int64)) if return_actor_ids else empty
        corners_arr = np.stack(all_corners_ego, axis=0)
        if return_actor_ids:
            return corners_arr, np.asarray(all_actor_ids, dtype=np.int64)
        return corners_arr

    def save(self, out_path: str, extra_meta: Optional[Dict[str, Any]] = None):
        """Compute final AP and write JSON."""
        _, _, calculate_ap, _, _ = _lazy_eval_imports()
        ap_table: Dict[str, Any] = {}
        for iou in self.iou_thresholds:
            n_tp = int(np.sum(self.result_stat[iou]["tp"]))
            n_fp = int(np.sum(self.result_stat[iou]["fp"]))
            n_gt = int(self.result_stat[iou]["gt"])
            # calculate_ap divides by gt and by precision-curve area; both blow up on empty data.
            if n_gt == 0 or (n_tp + n_fp) == 0:
                ap_value = 0.0
            else:
                try:
                    ap_value, _, _ = calculate_ap(self.result_stat, iou)
                    ap_value = float(ap_value)
                except ZeroDivisionError:
                    ap_value = 0.0
            ap_table[f"ap_{int(iou*100):02d}"] = {
                "ap":   ap_value,
                "iou":  float(iou),
                "n_tp": n_tp,
                "n_fp": n_fp,
                "n_gt": n_gt,
            }
        summary = {
            "n_frames":   self.n_frames,
            "cum_n_pred": self.cum_n_pred,
            "cum_n_pred_after_self_filter": self.cum_n_pred_after_self_filter,
            "cum_n_self_dets": self.cum_n_self_dets,
            "cum_n_gt":   self.cum_n_gt,
            "gt_filter": {
                "enabled": self.visible_ids_per_frame is not None,
                "cum_frames_with_filter": self.cum_frames_with_filter,
                "cum_frames_filter_empty": self.cum_frames_filter_empty,
                "n_frames_indexed": (
                    len(self.visible_ids_per_frame)
                    if self.visible_ids_per_frame is not None else 0
                ),
            },
            "ap":         ap_table,
        }
        if extra_meta:
            summary.update(extra_meta)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)


def _idle_control():
    c = carla.VehicleControl()
    c.steer = 0.0
    c.throttle = 0.0
    c.brake = 0.0
    c.hand_brake = False
    c.manual_gear_shift = False
    return c
