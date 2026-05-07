"""Stage internals: dataset assembly, HTML rendering, and CLI orchestration."""

from __future__ import annotations

import time

from v2xpnp.pipeline import runtime_common as _s1
from v2xpnp.pipeline import runtime_projection as _s2
from v2xpnp.pipeline import runtime_postprocess as _s3

for _mod in (_s1, _s2, _s3):
    for _name, _value in vars(_mod).items():
        if _name.startswith("__"):
            continue
        globals()[_name] = _value

def load_trajectories(
    yaml_dirs: Sequence[Path],
    dt: float,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
    yaw_deg: float = 0.0,
    flip_y: bool = False,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    List[List[Waypoint]],
    List[List[float]],
    Dict[int, Dict[str, object]],
]:
    """
    Load trajectories from YAML directories.
    
    Args:
        yaml_dirs: List of YAML directories to process
        dt: Time step between frames
        tx, ty, tz: Translation offsets
        yaw_deg: Rotation offset in degrees
        flip_y: Whether to flip Y coordinates
        
    Returns:
        vehicles: Dict mapping vehicle ID to list of waypoints
        vehicle_times: Dict mapping vehicle ID to list of timestamps
        ego_trajs: List of ego trajectories (one per source)
        ego_times: List of ego time lists
        obj_info: Dict mapping vehicle ID to metadata
    """
    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_trajs: List[List[Waypoint]] = []
    ego_times: List[List[float]] = []
    obj_info: Dict[int, Dict[str, object]] = {}
    actor_source_subdir: Dict[int, str] = {}
    actor_orig_vid: Dict[int, int] = {}

    def _transform_wp(wp: Waypoint) -> Waypoint:
        x, y = apply_se2(
            (float(wp.x), float(wp.y)),
            float(yaw_deg),
            float(tx),
            float(ty),
            flip_y=bool(flip_y),
        )
        yaw = float(wp.yaw)
        if bool(flip_y):
            yaw = -yaw
        yaw += float(yaw_deg)
        return Waypoint(
            x=float(x),
            y=float(y),
            z=float(getattr(wp, "z", 0.0)) + float(tz),
            yaw=float(yaw),
            pitch=float(getattr(wp, "pitch", 0.0)),
            roll=float(getattr(wp, "roll", 0.0)),
        )

    def _transform_traj(traj: Sequence[Waypoint]) -> List[Waypoint]:
        return [_transform_wp(wp) for wp in traj]

    # Prefer robust same-ID subdir merge from yaml_to_map to avoid overlapping
    # append artifacts when the same actor appears in multiple source subdirs.
    use_robust_merge = _env_int("V2X_USE_ROBUST_SUBDIR_MERGE", 1, minimum=0, maximum=1) == 1
    id_merge_distance_m = _env_float("V2X_ID_MERGE_DISTANCE_M", 8.0)
    if use_robust_merge and hasattr(ytm, "_merge_subdir_trajectories"):
        try:
            (
                v_map,
                v_times,
                merged_ego_trajs,
                merged_ego_times,
                v_info,
                actor_source_subdir,
                actor_orig_vid,
                merge_stats,
            ) = ytm._merge_subdir_trajectories(
                yaml_dirs=yaml_dirs,
                dt=float(dt),
                id_merge_distance_m=float(id_merge_distance_m),
            )

            v_map, v_times, v_info = _apply_overlap_dedup_pipeline(
                vehicles=v_map,
                vehicle_times=v_times,
                ego_trajs=merged_ego_trajs,
                ego_times=merged_ego_times,
                obj_info=v_info,
                actor_source_subdir=actor_source_subdir,
                actor_orig_vid=actor_orig_vid,
                dt=float(dt),
            )

            for vid, traj in v_map.items():
                if not traj:
                    continue
                vehicles[int(vid)] = _transform_traj(traj)
                vehicle_times[int(vid)] = [float(t) for t in v_times.get(vid, [])]
            for traj, times in zip(merged_ego_trajs, merged_ego_times):
                if not traj:
                    continue
                ego_trajs.append(_transform_traj(traj))
                ego_times.append([float(t) for t in times])
            for vid, meta in v_info.items():
                obj_info[int(vid)] = dict(meta)

            ids_with_collisions = _safe_int(merge_stats.get("ids_with_collisions", 0), 0) if isinstance(merge_stats, dict) else 0
            merged_duplicates = _safe_int(merge_stats.get("merged_duplicates", 0), 0) if isinstance(merge_stats, dict) else 0
            split_tracks_created = _safe_int(merge_stats.get("split_tracks_created", 0), 0) if isinstance(merge_stats, dict) else 0
            if ids_with_collisions > 0:
                print(
                    "[INFO] Robust subdir merge: ids_with_collisions={} merged_duplicates={} split_tracks_created={} id_merge_distance_m={:.2f}".format(
                        int(ids_with_collisions),
                        int(merged_duplicates),
                        int(split_tracks_created),
                        float(id_merge_distance_m),
                    )
                )
            return vehicles, vehicle_times, ego_trajs, ego_times, obj_info
        except Exception as exc:
            print(f"[WARN] Robust subdir merge unavailable; falling back to legacy append merge: {exc}")

    # Legacy merge path (append across yaml_dirs).
    for yd in yaml_dirs:
        is_negative = _is_negative_subdir(yd)
        v_map, v_times, ego_traj, ego_time, v_info = build_trajectories(
            yaml_dir=yd,
            dt=float(dt),
            tx=float(tx),
            ty=float(ty),
            tz=float(tz),
            yaw_deg=float(yaw_deg),
            flip_y=bool(flip_y),
        )

        if ego_traj and not is_negative:
            ego_trajs.append(ego_traj)
            ego_times.append(ego_time)

        for vid, traj in v_map.items():
            if not traj:
                continue
            if vid in vehicles:
                vehicles[vid].extend(traj)
                vehicle_times[vid].extend(v_times.get(vid, []))
            else:
                vehicles[vid] = list(traj)
                vehicle_times[vid] = list(v_times.get(vid, []))
                actor_source_subdir[int(vid)] = str(yd.name)
                actor_orig_vid[int(vid)] = int(vid)

            if vid in v_info:
                if vid not in obj_info:
                    obj_info[vid] = {}
                for k, v in v_info[vid].items():
                    if k not in obj_info[vid] or obj_info[vid][k] is None:
                        obj_info[vid][k] = v

    vehicles, vehicle_times, obj_info = _apply_overlap_dedup_pipeline(
        vehicles=vehicles,
        vehicle_times=vehicle_times,
        ego_trajs=ego_trajs,
        ego_times=ego_times,
        obj_info=obj_info,
        actor_source_subdir=actor_source_subdir,
        actor_orig_vid=actor_orig_vid,
        dt=float(dt),
    )

    return vehicles, vehicle_times, ego_trajs, ego_times, obj_info


# =============================================================================
# HTML Generation
# =============================================================================


def _build_html(dataset: Dict[str, object], multi_mode: bool = False) -> str:
    """Build interactive HTML visualization with optional multi-scenario support."""
    dataset_json = json.dumps(_sanitize_for_json(dataset), ensure_ascii=True, separators=(",", ":"))

    comparison_mode = False
    comparison_row = dataset.get("comparison_mode")
    if isinstance(comparison_row, dict):
        comparison_mode = str(comparison_row.get("kind", "")).strip().lower() == "profiles"

    # Scenario selector HTML (only shown in multi mode)
    scenario_selector_html = ""
    if multi_mode:
        selector_title = "Variant" if comparison_mode else "Scenario"
        scenario_selector_html = """
      <div class="section">
        <h3>""" + selector_title + """</h3>
        <div class="row">
          <select id="scenarioSelect" style="flex:1; min-height:32px; border:1px solid #365d75; border-radius:6px; background:#0f2433; color:var(--text); padding:4px 8px;"></select>
        </div>
        <div class="mono" id="scenarioInfo">-</div>
      </div>"""

    best_pick_html = ""
    if comparison_mode:
        best_pick_html = """
      <div class="section">
        <h3>Pick Best Variant</h3>
        <div class="row">
          <button id="markBestBtn">Mark Current as Best</button>
          <button id="clearBestBtn">Clear Best</button>
        </div>
        <div class="mono" id="bestPickInfo">No best variant selected.</div>
      </div>"""
    
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trajectory Visualization</title>
  <style>
    :root {{
      --bg: #0d1a24;
      --panel: #142635;
      --border: #2a4a60;
      --text: #e4edf3;
      --muted: #a8bccb;
      --accent: #4ec4ff;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; background: var(--bg); color: var(--text); font-family: "Segoe UI", sans-serif; }}
    #app {{ height: 100%; display: grid; grid-template-columns: 1fr 340px; gap: 10px; padding: 10px; }}
    #main {{ position: relative; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #0b1822; }}
    #canvas {{ width: 100%; height: 100%; display: block; }}
    #hud {{ position: absolute; left: 10px; top: 10px; background: rgba(7, 16, 24, 0.85); border: 1px solid #365a72; border-radius: 8px; padding: 8px 10px; font-size: 12px; }}
    #hud .line {{ margin: 2px 0; }}
    #sidebar {{ border: 1px solid var(--border); border-radius: 8px; background: var(--panel); padding: 12px; overflow: auto; }}
    h3 {{ margin: 10px 0 6px 0; font-size: 13px; color: var(--accent); }}
    .section {{ border: 1px solid #29465c; border-radius: 8px; padding: 8px; margin-bottom: 10px; background: rgba(9, 18, 26, 0.5); }}
    .row {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
    button {{ min-height: 32px; border: 1px solid #365d75; border-radius: 6px; background: linear-gradient(180deg, #2e5068 0%, #253f53 100%); color: var(--text); cursor: pointer; flex: 1; }}
    button:hover {{ border-color: #4e89a9; }}
    input[type="range"] {{ flex: 1; }}
    .legendItem {{ display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 4px; }}
    .legendSwatch {{ width: 14px; height: 8px; border-radius: 2px; }}
    .markerLegendRow {{ display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:4px; }}
    .markerSwatch {{ width:12px; height:12px; display:inline-block; }}
    .markerRaw {{ border-radius:50%; background:#8fb3cc; border:1px solid #0b1822; }}
    .markerV2 {{ background:#ffffff; border:2px solid #ffb347; }}
    .markerCarlaPre {{ background:#c78cff; border-radius:2px; border:1px solid #f4d9ff; }}
    .markerCarla {{ background:#58ecff; transform:rotate(45deg); border:1px solid #e8fdff; }}
    .markerEgo {{ border-radius:50%; border:2px solid #f8c65f; box-shadow:0 0 0 2px rgba(248,198,95,0.35); background:rgba(248,198,95,0.18); }}
    .mono {{ font-family: monospace; font-size: 11px; }}
    #actorList {{ max-height: 180px; overflow: auto; border: 1px solid #2f4e63; border-radius: 6px; padding: 6px; background: rgba(7, 14, 21, 0.5); }}
    .actorRow {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 11px; padding: 2px 4px; border-radius: 4px; cursor: pointer; }}
    .actorRow.ego {{ border: 1px solid rgba(248, 198, 95, 0.35); background: rgba(248, 198, 95, 0.08); }}
    .actorRow:hover {{ background: rgba(78, 196, 255, 0.14); }}
    .actorRow.active {{ background: rgba(78, 196, 255, 0.26); border: 1px solid #4ec4ff; }}
    .actorDot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    .actorBadge {{ display: inline-flex; align-items: center; justify-content: center; min-width: 26px; height: 14px; padding: 0 4px; border-radius: 10px; font-size: 9px; font-weight: 700; letter-spacing: 0.3px; }}
    .actorBadge.ego {{ background: rgba(248, 198, 95, 0.28); color: #ffe9a8; border: 1px solid rgba(248, 198, 95, 0.7); }}
    .actorBadge.issue {{ background: rgba(255, 108, 108, 0.24); color: #ffd2d2; border: 1px solid rgba(255, 122, 122, 0.75); }}
    #anomalyList {{ max-height: 180px; overflow: auto; border: 1px solid #5c2f2f; border-radius: 6px; padding: 6px; background: rgba(26, 11, 14, 0.45); }}
    .anomalyRow {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 11px; padding: 3px 4px; border-radius: 4px; border: 1px solid rgba(140, 53, 53, 0.45); cursor: pointer; }}
    .anomalyRow:hover {{ background: rgba(255, 120, 120, 0.1); }}
    .anomalyRow.active {{ background: rgba(255, 120, 120, 0.18); border-color: rgba(255, 145, 145, 0.8); }}
    .anomalyDot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    .anomalyLow {{ background: #f6d36f; }}
    .anomalyMid {{ background: #ff9f57; }}
    .anomalyHigh {{ background: #ff5f5f; }}
  </style>
</head>
<body>
  <div id="app">
    <div id="main">
      <canvas id="canvas"></canvas>
      <div id="hud">
        <div class="line" id="hudScenario">Scenario: -</div>
        <div class="line" id="hudMap">Map: -</div>
        <div class="line" id="hudTime">Time: -</div>
        <div class="line" id="hudCounts">Actors: -</div>
      </div>
    </div>
    <aside id="sidebar">
      <h3>Trajectory Visualization</h3>
      {scenario_selector_html}
      {best_pick_html}
      <div class="section">
        <h3>Timeline</h3>
        <div class="row">
          <button id="playBtn">Play</button>
          <button id="fitBtn">Fit View</button>
        </div>
        <div class="row">
          <input id="timeSlider" type="range" min="0" max="0" value="0" step="1" />
        </div>
        <div class="mono" id="timeLabel">t=0.000s</div>
      </div>

      <div class="section">
        <h3>Lane Types</h3>
        <div id="laneLegend"></div>
      </div>

      <div class="section">
        <h3>Display</h3>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showRawLayerToggle" checked />
          Show raw trajectory layer
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showAlignedToggle" checked />
          Show V2XPNP snapped layer
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showCarlaPreToggle" checked />
          Show CARLA pre-postprocess layer
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showCarlaProjectedToggle" checked />
          Show CARLA final layer
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showCarlaMapToggle" checked />
          Show CARLA lane map layer
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showCarlaDirectionToggle" checked />
          Show CARLA lane direction arrows
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showCarlaImageToggle" checked />
          Show CARLA top-down image underlay
        </label>
        <div class="row" style="margin-bottom:6px;">
          <span class="mono" style="min-width:94px;">Image opacity</span>
          <input id="carlaImageOpacitySlider" type="range" min="0" max="100" step="1" value="55" />
          <span class="mono" id="carlaImageOpacityLabel">55%</span>
        </div>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showDeltaVectorsToggle" checked />
          Show layer delta connectors
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showPostShiftHotspotsToggle" checked />
          Highlight strong CARLA post edits
        </label>
        <div class="row" style="margin-bottom:6px;">
          <span class="mono" style="min-width:94px;">Post edit min</span>
          <input id="postShiftMinSlider" type="range" min="0" max="80" step="1" value="15" />
          <span class="mono" id="postShiftMinLabel">1.5m</span>
        </div>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showTrajToggle" checked />
          Show full trajectory paths
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showBothDotsToggle" checked />
          Show comparison dots (raw / V2 / pre / final)
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="dimNonFocusedToggle" checked />
          Dim non-focused actors
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showLaneLabelsToggle" />
          Show lane_id labels on map
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showActorIdToggle" checked />
          Show actor IDs next to actors
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showActorLaneToggle" checked />
          Show assigned lane_id next to actors
        </label>
        <div class="mono" id="alignInfo" style="margin-top:6px; color:#a8bccb;">-</div>
      </div>

      <div class="section">
        <h3>Marker Key</h3>
        <div class="markerLegendRow">
          <span class="markerSwatch markerRaw"></span>
          Raw trajectory marker (circle, role color)
        </div>
        <div class="markerLegendRow">
          <span class="markerSwatch markerV2"></span>
          V2XPNP aligned marker (square, orange outline)
        </div>
        <div class="markerLegendRow">
          <span class="markerSwatch markerCarlaPre"></span>
          CARLA pre-postprocess marker (violet square)
        </div>
        <div class="markerLegendRow">
          <span class="markerSwatch markerCarla"></span>
          CARLA projected marker (diamond, cyan)
        </div>
        <div class="markerLegendRow">
          <span class="markerSwatch markerEgo"></span>
          Ego highlight halo (gold ring)
        </div>
        <div class="mono" style="color:#ffb5b5;">Red rings = strong CARLA pre→final postprocess edits.</div>
        <div class="mono" style="color:#a8bccb;">Focused actor shows marker tags: R / V / P / C.</div>
      </div>

      <div class="section">
        <h3>Anomaly Finder</h3>
        <div class="row">
          <button id="prevIssueBtn">Prev Issue</button>
          <button id="nextIssueBtn">Next Issue</button>
        </div>
        <div class="row">
          <button id="focusWorstIssueBtn">Focus Worst Actor</button>
        </div>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showAnomaliesToggle" checked />
          Highlight anomalies on map
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showOnlyAnomaliesToggle" />
          Show only actors with anomalies
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="autoFocusIssueToggle" checked />
          Auto-focus actor on issue jump
        </label>
        <div class="row" style="margin-bottom:6px;">
          <span class="mono" style="min-width:94px;">Min severity</span>
          <input id="anomalyMinSlider" type="range" min="1" max="5" step="0.5" value="2" />
          <span class="mono" id="anomalyMinLabel">2.0</span>
        </div>
        <div class="mono" id="anomalySummary" style="margin-bottom:6px; color:#ffbcbc;">-</div>
        <div id="anomalyList"></div>
      </div>

      <div class="section">
        <h3>Actors</h3>
        <div class="row">
          <button id="focusEgoBtn">Focus First Ego</button>
          <button id="cycleEgoBtn">Next Ego</button>
        </div>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="highlightEgoToggle" checked />
          Highlight ego markers
        </label>
        <label style="display:flex; align-items:center; gap:8px; font-size:12px; margin-bottom:6px;">
          <input type="checkbox" id="showEgoOnlyToggle" />
          Show ego actors only
        </label>
        <div class="row">
          <button id="clearFocusBtn">Clear Focus</button>
        </div>
        <div class="mono" style="margin-bottom:6px; color:#a8bccb;">Click an actor row to focus (good for inspecting divergence).</div>
        <div id="actorLegend"></div>
        <div id="actorList"></div>
      </div>
    </aside>
  </div>

  <script id="dataset" type="application/json">{dataset_json}</script>
  <script>
  (() => {{
    'use strict';

    const RAW_DATA = JSON.parse(document.getElementById('dataset').textContent);
    const MULTI_MODE = {'true' if multi_mode else 'false'};
    const COMPARISON_MODE = {'true' if comparison_mode else 'false'};
    
    // In multi-mode, RAW_DATA.scenarios is an array; in single mode, RAW_DATA is the scenario itself
    const scenarios = MULTI_MODE ? (RAW_DATA.scenarios || []) : [RAW_DATA];
    const comparisonMeta = (RAW_DATA && typeof RAW_DATA === 'object' && RAW_DATA.comparison_mode && typeof RAW_DATA.comparison_mode === 'object')
      ? RAW_DATA.comparison_mode
      : {{}};
    const GLOBAL_CARLA_IMAGES = (RAW_DATA && typeof RAW_DATA === 'object' && RAW_DATA.carla_images && typeof RAW_DATA.carla_images === 'object')
      ? RAW_DATA.carla_images
      : {{}};
    let currentScenarioIdx = 0;
    
    function getCurrentData() {{
      return scenarios[currentScenarioIdx] || {{}};
    }}

    function getProfileName(data, fallbackIdx = null) {{
      const row = data && typeof data === 'object' ? data.processing_profile : null;
      const name = row && typeof row === 'object' ? String(row.name || '').trim() : '';
      if (name) return name;
      if (COMPARISON_MODE) {{
        const idx = Number.isFinite(Number(fallbackIdx)) ? Number(fallbackIdx) : currentScenarioIdx;
        return 'variant_' + String(Math.max(0, Math.round(idx)) + 1);
      }}
      return '';
    }}

    function getComparisonStorageKey() {{
      const baseScenario = String(
        (comparisonMeta && comparisonMeta.base_scenario)
          || (scenarios[0] && scenarios[0].base_scenario_name)
          || (scenarios[0] && scenarios[0].scenario_name)
          || 'scenario'
      );
      const profiles = Array.isArray(comparisonMeta && comparisonMeta.profiles)
        ? comparisonMeta.profiles.map(v => String(v || '').trim()).filter(Boolean).join(',')
        : '';
      return 'trajviz_best_variant::' + baseScenario + '::' + profiles;
    }}

    function loadBestVariantProfile() {{
      if (!COMPARISON_MODE) return '';
      try {{
        const key = getComparisonStorageKey();
        const stored = String(window.localStorage.getItem(key) || '').trim();
        return stored;
      }} catch (err) {{
        return '';
      }}
    }}

    function saveBestVariantProfile(profileName) {{
      if (!COMPARISON_MODE) return;
      const name = String(profileName || '').trim();
      try {{
        const key = getComparisonStorageKey();
        if (name) window.localStorage.setItem(key, name);
        else window.localStorage.removeItem(key);
      }} catch (err) {{
        // Ignore storage failures in restricted browser contexts.
      }}
      state.bestVariantProfile = name;
    }}

    function refreshScenarioSelectorLabels() {{
      if (!MULTI_MODE || !scenarioSelect) return;
      for (let i = 0; i < scenarios.length; i++) {{
        const opt = scenarioSelect.options[i];
        if (!opt) continue;
        const row = scenarios[i] || {{}};
        if (COMPARISON_MODE) {{
          const profile = getProfileName(row, i) || ('variant_' + String(i + 1));
          const isBest = !!(state.bestVariantProfile && profile === state.bestVariantProfile);
          opt.textContent = (isBest ? '★ ' : '') + profile;
        }} else {{
          opt.textContent = row.scenario_name || `Scenario ${{i + 1}}`;
        }}
      }}
    }}

    function updateBestPickPanel() {{
      if (!COMPARISON_MODE) return;
      const currentProfile = getProfileName(getCurrentData());
      const selected = String(state.bestVariantProfile || '').trim();
      const isCurrentBest = !!(selected && currentProfile && selected === currentProfile);
      if (bestPickInfo) {{
        if (selected) {{
          bestPickInfo.textContent = 'Best variant: ' + selected + (isCurrentBest ? ' (current)' : '');
        }} else {{
          bestPickInfo.textContent = 'No best variant selected.';
        }}
      }}
      if (markBestBtn) {{
        markBestBtn.disabled = !currentProfile;
      }}
    }}

    const laneTypePalette = {{
      '1': '#4e79a7',
      '2': '#f28e2b',
      '3': '#59a14f',
      '4': '#e15759',
      'unknown': '#9aa6af',
    }};
    const actorRolePalette = {{
      ego: '#f8c65f',
      vehicle: '#6bc6ff',
      walker: '#7df0a8',
      cyclist: '#c3a4ff',
      npc: '#6bc6ff',
    }};

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const hudScenario = document.getElementById('hudScenario');
    const hudMap = document.getElementById('hudMap');
    const hudTime = document.getElementById('hudTime');
    const hudCounts = document.getElementById('hudCounts');
    const slider = document.getElementById('timeSlider');
    const playBtn = document.getElementById('playBtn');
    const fitBtn = document.getElementById('fitBtn');
    const timeLabel = document.getElementById('timeLabel');
    const laneLegend = document.getElementById('laneLegend');
    const actorLegend = document.getElementById('actorLegend');
    const actorList = document.getElementById('actorList');
    const clearFocusBtn = document.getElementById('clearFocusBtn');
    const focusEgoBtn = document.getElementById('focusEgoBtn');
    const cycleEgoBtn = document.getElementById('cycleEgoBtn');
    const highlightEgoToggle = document.getElementById('highlightEgoToggle');
    const showEgoOnlyToggle = document.getElementById('showEgoOnlyToggle');
    const scenarioSelect = document.getElementById('scenarioSelect');
    const scenarioInfo = document.getElementById('scenarioInfo');
    const markBestBtn = document.getElementById('markBestBtn');
    const clearBestBtn = document.getElementById('clearBestBtn');
    const bestPickInfo = document.getElementById('bestPickInfo');
    const showRawLayerToggle = document.getElementById('showRawLayerToggle');
    const showAlignedToggle = document.getElementById('showAlignedToggle');
    const showCarlaPreToggle = document.getElementById('showCarlaPreToggle');
    const showCarlaProjectedToggle = document.getElementById('showCarlaProjectedToggle');
    const showCarlaMapToggle = document.getElementById('showCarlaMapToggle');
    const showCarlaDirectionToggle = document.getElementById('showCarlaDirectionToggle');
    const showCarlaImageToggle = document.getElementById('showCarlaImageToggle');
    const carlaImageOpacitySlider = document.getElementById('carlaImageOpacitySlider');
    const carlaImageOpacityLabel = document.getElementById('carlaImageOpacityLabel');
    const showDeltaVectorsToggle = document.getElementById('showDeltaVectorsToggle');
    const showPostShiftHotspotsToggle = document.getElementById('showPostShiftHotspotsToggle');
    const postShiftMinSlider = document.getElementById('postShiftMinSlider');
    const postShiftMinLabel = document.getElementById('postShiftMinLabel');
    const showTrajToggle = document.getElementById('showTrajToggle');
    const showBothDotsToggle = document.getElementById('showBothDotsToggle');
    const dimNonFocusedToggle = document.getElementById('dimNonFocusedToggle');
    const showLaneLabelsToggle = document.getElementById('showLaneLabelsToggle');
    const showActorIdToggle = document.getElementById('showActorIdToggle');
    const showActorLaneToggle = document.getElementById('showActorLaneToggle');
    const alignInfo = document.getElementById('alignInfo');
    const prevIssueBtn = document.getElementById('prevIssueBtn');
    const nextIssueBtn = document.getElementById('nextIssueBtn');
    const focusWorstIssueBtn = document.getElementById('focusWorstIssueBtn');
    const showAnomaliesToggle = document.getElementById('showAnomaliesToggle');
    const showOnlyAnomaliesToggle = document.getElementById('showOnlyAnomaliesToggle');
    const autoFocusIssueToggle = document.getElementById('autoFocusIssueToggle');
    const anomalyMinSlider = document.getElementById('anomalyMinSlider');
    const anomalyMinLabel = document.getElementById('anomalyMinLabel');
    const anomalySummary = document.getElementById('anomalySummary');
    const anomalyList = document.getElementById('anomalyList');

    const state = {{
      tIndex: 0,
      playing: false,
      playHandle: null,
      view: {{ cx: 0, cy: 0, scale: 2.5 }},
      drag: null,
      showRawLayer: true,
      showAligned: true,
      showCarlaPre: true,
      showCarlaProjected: true,
      showCarlaMap: true,
      showCarlaDirection: true,
      showCarlaImage: true,
      carlaImageOpacity: 0.55,
      showDeltaVectors: true,
      showPostShiftHotspots: true,
      postShiftMinM: 1.5,
      showTraj: true,
      showBothDots: true,
      dimNonFocused: true,
      showLaneLabels: false,
      showActorId: true,
      showActorLane: true,
      highlightEgo: true,
      showEgoOnly: false,
      showAnomalies: true,
      showOnlyAnomalies: false,
      autoFocusIssue: true,
      anomalyMinSeverity: 2.0,
      focusActorId: '',
      lastFocusedEgoIdx: -1,
      anomalyCursor: -1,
      anomalyData: null,
      bestVariantProfile: '',
    }};
    const carlaImageCache = new Map();

    function isEgoTrack(track) {{
      return String((track && track.role) || '') === 'ego';
    }}

    function sortTracksForList(tracks) {{
      return [...tracks].sort((a, b) => {{
        const egoDelta = (isEgoTrack(a) ? 0 : 1) - (isEgoTrack(b) ? 0 : 1);
        if (egoDelta !== 0) return egoDelta;
        const aNum = Number(a.id);
        const bNum = Number(b.id);
        if (Number.isFinite(aNum) && Number.isFinite(bNum) && aNum !== bNum) return aNum - bNum;
        return String(a.id).localeCompare(String(b.id));
      }});
    }}

    function getEgoActorIds(tracks) {{
      return sortTracksForList(tracks)
        .filter(track => isEgoTrack(track))
        .map(track => String(track.id));
    }}

    function focusFirstEgo() {{
      const DATA = getCurrentData();
      const egoIds = getEgoActorIds(DATA.tracks || []);
      if (!egoIds.length) return;
      state.focusActorId = egoIds[0];
      state.lastFocusedEgoIdx = 0;
      updateActorList();
      render();
    }}

    function cycleEgoFocus() {{
      const DATA = getCurrentData();
      const egoIds = getEgoActorIds(DATA.tracks || []);
      if (!egoIds.length) return;
      let idx = egoIds.indexOf(String(state.focusActorId));
      if (idx < 0) idx = state.lastFocusedEgoIdx;
      idx = (idx + 1 + egoIds.length) % egoIds.length;
      state.focusActorId = egoIds[idx];
      state.lastFocusedEgoIdx = idx;
      updateActorList();
      render();
    }}

    function _safeNum(v, fallback = 0.0) {{
      const n = Number(v);
      return Number.isFinite(n) ? n : Number(fallback);
    }}

    function _isValidBounds(bounds) {{
      if (!bounds || typeof bounds !== 'object') return false;
      const minX = Number(bounds.min_x);
      const maxX = Number(bounds.max_x);
      const minY = Number(bounds.min_y);
      const maxY = Number(bounds.max_y);
      return Number.isFinite(minX) && Number.isFinite(maxX) && Number.isFinite(minY) && Number.isFinite(maxY)
        && (maxX > minX) && (maxY > minY);
    }}

    function _resolveCarlaImageMeta(DATA) {{
      const carlaMap = DATA && typeof DATA === 'object' ? DATA.carla_map : null;
      if (!carlaMap || typeof carlaMap !== 'object') return null;

      if (typeof carlaMap.image_b64 === 'string' && carlaMap.image_b64.length > 0 && _isValidBounds(carlaMap.image_bounds)) {{
        return {{
          key: 'inline:' + String(currentScenarioIdx),
          b64: String(carlaMap.image_b64),
          bounds: carlaMap.image_bounds,
        }};
      }}

      const ref = String(carlaMap.image_ref || '').trim();
      if (!ref) return null;
      const globalRow = GLOBAL_CARLA_IMAGES && typeof GLOBAL_CARLA_IMAGES === 'object'
        ? GLOBAL_CARLA_IMAGES[ref]
        : null;
      if (!globalRow || typeof globalRow !== 'object') return null;
      const b64 = String(globalRow.image_b64 || '');
      const bounds = _isValidBounds(globalRow.image_bounds) ? globalRow.image_bounds : (_isValidBounds(carlaMap.image_bounds) ? carlaMap.image_bounds : null);
      if (!b64 || !bounds) return null;
      return {{
        key: 'ref:' + ref,
        b64: b64,
        bounds: bounds,
      }};
    }}

    function _ensureCarlaImageLoaded(DATA) {{
      const meta = _resolveCarlaImageMeta(DATA);
      if (!meta) return null;
      const key = String(meta.key);
      let row = carlaImageCache.get(key);
      if (row) return row;
      const img = new Image();
      row = {{
        img: img,
        ready: false,
        bounds: meta.bounds,
      }};
      img.onload = () => {{
        row.ready = true;
        render();
      }};
      img.src = 'data:image/jpeg;base64,' + meta.b64;
      carlaImageCache.set(key, row);
      return row;
    }}

    function _syncCarlaImageControls() {{
      const DATA = getCurrentData();
      const hasImage = !!_resolveCarlaImageMeta(DATA);
      if (showCarlaImageToggle) {{
        showCarlaImageToggle.disabled = !hasImage;
        if (!hasImage) {{
          showCarlaImageToggle.checked = false;
          state.showCarlaImage = false;
        }} else {{
          if (!showCarlaImageToggle.checked && !state.showCarlaImage) {{
            showCarlaImageToggle.checked = true;
            state.showCarlaImage = true;
          }}
        }}
      }}
      if (carlaImageOpacitySlider) {{
        carlaImageOpacitySlider.disabled = !hasImage;
      }}
      if (carlaImageOpacityLabel) {{
        const pct = Math.round(Math.max(0, Math.min(100, Number(state.carlaImageOpacity) * 100)));
        carlaImageOpacityLabel.textContent = String(pct) + '%';
      }}
      if (hasImage) _ensureCarlaImageLoaded(DATA);
    }}

    function _hasFiniteXY(frame, xKey, yKey) {{
      return Number.isFinite(Number(frame?.[xKey])) && Number.isFinite(Number(frame?.[yKey]));
    }}

    function _yawNormDeg(y) {{
      let v = _safeNum(y, 0.0);
      while (v > 180.0) v -= 360.0;
      while (v <= -180.0) v += 360.0;
      return v;
    }}

    function _yawDiffDeg(a, b) {{
      return Math.abs(_yawNormDeg(_safeNum(a, 0.0) - _safeNum(b, 0.0)));
    }}

    function _laneKey(frame) {{
      if (!frame || typeof frame !== 'object') return '';
      if (Number.isFinite(Number(frame.assigned_lane_id))) return 'L' + String(Math.round(Number(frame.assigned_lane_id)));
      if (Number.isFinite(Number(frame.lane_id))) return 'L' + String(Math.round(Number(frame.lane_id)));
      return '';
    }}

    function _eventKey(ev) {{
      if (!ev || typeof ev !== 'object') return '';
      return String(ev.trackId) + '|' + String(ev.frameIndex) + '|' + String(ev.type);
    }}

    function _severityClass(severity) {{
      const sev = _safeNum(severity, 0.0);
      if (sev >= 4.0) return 'anomalyHigh';
      if (sev >= 2.5) return 'anomalyMid';
      return 'anomalyLow';
    }}

    function _nearestTimelineIndex(targetTime) {{
      const timeline = getTimeline();
      if (!timeline.length) return 0;
      let best = 0;
      let bestDiff = Math.abs(_safeNum(timeline[0], 0.0) - _safeNum(targetTime, 0.0));
      for (let i = 1; i < timeline.length; i++) {{
        const diff = Math.abs(_safeNum(timeline[i], 0.0) - _safeNum(targetTime, 0.0));
        if (diff < bestDiff) {{
          bestDiff = diff;
          best = i;
        }}
      }}
      return best;
    }}

    function _computeTrackAnomalies(track) {{
      const frames = Array.isArray(track?.frames) ? track.frames : [];
      const role = String(track?.role || '');
      const actorId = String(track?.id || '');
      const out = {{
        id: actorId,
        role: role,
        score: 0.0,
        maxSeverity: 0.0,
        laneChanges: 0,
        eventCount: 0,
        counts: {{}},
        events: [],
      }};
      if (frames.length < 2) return out;
      const plannerRole = (role === 'ego' || role === 'vehicle' || role === 'cyclist');
      if (!plannerRole) return out;

      const stepRaw = new Array(frames.length).fill(0.0);
      const stepDisplay = new Array(frames.length).fill(0.0);
      const stepCarla = new Array(frames.length).fill(0.0);
      const laneSeq = new Array(frames.length).fill('');
      const laneChangeIdx = [];
      const seen = new Map();

      const getDisplayXY = (f) => {{
        if (_hasFiniteXY(f, 'cx', 'cy')) return [Number(f.cx), Number(f.cy)];
        if (_hasFiniteXY(f, 'sx', 'sy')) return [Number(f.sx), Number(f.sy)];
        if (_hasFiniteXY(f, 'x', 'y')) return [Number(f.x), Number(f.y)];
        return null;
      }};

      const pushEvent = (frameIndex, type, severity, detail) => {{
        const fi = Math.max(0, Math.min(frames.length - 1, Math.round(_safeNum(frameIndex, 0))));
        const sev = Math.max(0.5, Math.min(5.0, _safeNum(severity, 1.0)));
        const key = String(fi) + '|' + String(type);
        if (seen.has(key)) {{
          const ei = seen.get(key);
          const prev = out.events[ei];
          if (prev && sev > _safeNum(prev.severity, 0.0)) {{
            prev.severity = sev;
            prev.detail = String(detail || prev.detail || '');
          }}
          return;
        }}
        const fr = frames[fi] || {{}};
        out.counts[type] = (_safeNum(out.counts[type], 0.0) + 1);
        out.events.push({{
          trackId: actorId,
          role: role,
          frameIndex: fi,
          t: _safeNum(fr.t, 0.0),
          type: String(type),
          severity: sev,
          detail: String(detail || ''),
        }});
        seen.set(key, out.events.length - 1);
      }};

      for (let i = 1; i < frames.length; i++) {{
        const prev = frames[i - 1];
        const curr = frames[i];
        laneSeq[i] = _laneKey(curr);

        const dt = Math.max(1e-3, _safeNum(curr?.t, 0.0) - _safeNum(prev?.t, 0.0));

        if (_hasFiniteXY(prev, 'x', 'y') && _hasFiniteXY(curr, 'x', 'y')) {{
          stepRaw[i] = Math.hypot(Number(curr.x) - Number(prev.x), Number(curr.y) - Number(prev.y));
        }}
        const prevDisplay = getDisplayXY(prev);
        const currDisplay = getDisplayXY(curr);
        const nextDisplay = (i + 1 < frames.length) ? getDisplayXY(frames[i + 1]) : null;
        if (prevDisplay && currDisplay) {{
          stepDisplay[i] = Math.hypot(currDisplay[0] - prevDisplay[0], currDisplay[1] - prevDisplay[1]);
        }}
        if (_hasFiniteXY(prev, 'cx', 'cy') && _hasFiniteXY(curr, 'cx', 'cy')) {{
          stepCarla[i] = Math.hypot(Number(curr.cx) - Number(prev.cx), Number(curr.cy) - Number(prev.cy));
        }}

        if (laneSeq[i] && laneSeq[i - 1] && laneSeq[i] !== laneSeq[i - 1]) {{
          laneChangeIdx.push(i);
        }}

        const rawStep = stepRaw[i];
        const dispStep = stepDisplay[i];
        const carStep = stepCarla[i];

        // Planner-likeness mode: avoid treating gross timing/spawn transients as
        // high-severity issues; focus high-severity signals on persistent motion artifacts.

        if (_hasFiniteXY(curr, 'sx', 'sy') && _hasFiniteXY(curr, 'x', 'y')) {{
          const d = Math.hypot(Number(curr.sx) - Number(curr.x), Number(curr.sy) - Number(curr.y));
          if (d > 4.0) pushEvent(i, 'divergence_v2', Math.min(2.3, 1.1 + (d - 4.0) / 4.0), 'raw→V2=' + d.toFixed(2) + 'm');
        }}
        if (_hasFiniteXY(curr, 'cx', 'cy') && _hasFiniteXY(curr, 'x', 'y')) {{
          const d = Math.hypot(Number(curr.cx) - Number(curr.x), Number(curr.cy) - Number(curr.y));
          if (d > 5.2) pushEvent(i, 'divergence_carla', Math.min(2.6, 1.2 + (d - 5.2) / 3.2), 'raw→CARLA=' + d.toFixed(2) + 'm');
        }}
        if (_hasFiniteXY(curr, 'cx', 'cy') && _hasFiniteXY(curr, 'sx', 'sy')) {{
          const d = Math.hypot(Number(curr.cx) - Number(curr.sx), Number(curr.cy) - Number(curr.sy));
          if (d > 5.8) pushEvent(i, 'layer_conflict', Math.min(2.5, 1.2 + (d - 5.8) / 3.0), 'V2→CARLA=' + d.toFixed(2) + 'm');
        }}

        if (_hasFiniteXY(curr, 'cx', 'cy')) {{
          let motionYaw = null;
          if (prevDisplay && nextDisplay) {{
            const dx = Number(nextDisplay[0]) - Number(prevDisplay[0]);
            const dy = Number(nextDisplay[1]) - Number(prevDisplay[1]);
            if (Math.hypot(dx, dy) > 0.4) motionYaw = Math.atan2(dy, dx) * 180.0 / Math.PI;
          }} else if (currDisplay && nextDisplay) {{
            const dx = Number(nextDisplay[0]) - Number(currDisplay[0]);
            const dy = Number(nextDisplay[1]) - Number(currDisplay[1]);
            if (Math.hypot(dx, dy) > 0.4) motionYaw = Math.atan2(dy, dx) * 180.0 / Math.PI;
          }} else if (prevDisplay && currDisplay) {{
            const dx = Number(currDisplay[0]) - Number(prevDisplay[0]);
            const dy = Number(currDisplay[1]) - Number(prevDisplay[1]);
            if (Math.hypot(dx, dy) > 0.4) motionYaw = Math.atan2(dy, dx) * 180.0 / Math.PI;
          }}
          if (motionYaw != null) {{
            const dyaw = _yawDiffDeg(motionYaw, _safeNum(curr?.cyaw, 0.0));
            if (dyaw > 179.0) {{
              pushEvent(i, 'direction_mismatch', 1.8, 'yaw mismatch=' + dyaw.toFixed(1));
            }}
          }}
        }}
      }}

      out.laneChanges = laneChangeIdx.length;
      laneSeq[0] = _laneKey(frames[0]);

      if (laneChangeIdx.length >= 8) {{
        const idx = laneChangeIdx[Math.min(laneChangeIdx.length - 1, 3)];
        pushEvent(idx, 'many_lane_changes', 1.9, 'lane changes=' + laneChangeIdx.length);
      }}
      for (let i = 0; i < laneChangeIdx.length; i++) {{
        let j = i;
        while (j < laneChangeIdx.length && (laneChangeIdx[j] - laneChangeIdx[i]) <= 20) j++;
        const c = j - i;
        if (c >= 5) {{
          const idx = laneChangeIdx[j - 1];
          pushEvent(idx, 'lane_churn', 1.9, c + ' changes in <=20 frames');
        }}
      }}

      const laneRuns = [];
      let rs = -1;
      let rk = '';
      for (let i = 0; i < laneSeq.length; i++) {{
        const key = laneSeq[i];
        if (!key) continue;
        if (rs < 0) {{
          rs = i;
          rk = key;
          continue;
        }}
        if (key !== rk) {{
          laneRuns.push({{ start: rs, end: i - 1, key: rk }});
          rs = i;
          rk = key;
        }}
      }}
      if (rs >= 0) laneRuns.push({{ start: rs, end: laneSeq.length - 1, key: rk }});
      for (let i = 1; i < laneRuns.length - 1; i++) {{
        const a = laneRuns[i - 1];
        const b = laneRuns[i];
        const c = laneRuns[i + 1];
        const bLen = b.end - b.start + 1;
        const mid = Math.floor((b.start + b.end) / 2);
        if (a.key === c.key && a.key !== b.key && bLen <= 1) {{
          pushEvent(mid, 'lane_flicker', 1.8, 'A→B→A lane flicker');
        }} else if (a.key !== c.key && b.key !== a.key && b.key !== c.key && bLen <= 1) {{
          pushEvent(mid, 'transition_spike', 1.8, 'single-frame transition spike');
        }}
      }}

      out.events.sort((a, b) => {{
        const sa = _safeNum(a?.severity, 0.0);
        const sb = _safeNum(b?.severity, 0.0);
        if (Math.abs(sb - sa) > 1e-6) return sb - sa;
        return _safeNum(a?.t, 0.0) - _safeNum(b?.t, 0.0);
      }});
      out.eventCount = out.events.length;
      out.maxSeverity = out.events.length ? out.events[0].severity : 0.0;
      out.score = out.events.reduce((acc, ev) => acc + _safeNum(ev?.severity, 0.0), 0.0) + 0.25 * out.laneChanges;
      return out;
    }}

    function computeScenarioAnomalies(DATA) {{
      if (DATA && typeof DATA === 'object' && DATA.__anomalyCache) return DATA.__anomalyCache;
      const tracks = Array.isArray(DATA?.tracks) ? DATA.tracks : [];
      const byActor = {{}};
      const events = [];
      for (const track of tracks) {{
        const row = _computeTrackAnomalies(track);
        if (_safeNum(row?.eventCount, 0.0) <= 0) continue;
        byActor[String(row.id)] = row;
        for (const ev of row.events) events.push(ev);
      }}
      const actorOrder = Object.values(byActor).sort((a, b) => {{
        const ds = _safeNum(b?.score, 0.0) - _safeNum(a?.score, 0.0);
        if (Math.abs(ds) > 1e-6) return ds;
        const dm = _safeNum(b?.maxSeverity, 0.0) - _safeNum(a?.maxSeverity, 0.0);
        if (Math.abs(dm) > 1e-6) return dm;
        return String(a?.id || '').localeCompare(String(b?.id || ''));
      }});
      events.sort((a, b) => {{
        const ds = _safeNum(b?.severity, 0.0) - _safeNum(a?.severity, 0.0);
        if (Math.abs(ds) > 1e-6) return ds;
        return _safeNum(a?.t, 0.0) - _safeNum(b?.t, 0.0);
      }});
      const out = {{
        byActor: byActor,
        events: events,
        actorOrder: actorOrder,
        flaggedCount: actorOrder.length,
        totalTracks: tracks.length,
      }};
      if (DATA && typeof DATA === 'object') DATA.__anomalyCache = out;
      return out;
    }}

    function updateAnomalyPanel() {{
      const DATA = getCurrentData();
      state.anomalyData = computeScenarioAnomalies(DATA);
      const anomalyData = state.anomalyData || {{ byActor: {{}}, actorOrder: [], events: [], flaggedCount: 0, totalTracks: 0 }};
      if (anomalyMinLabel) anomalyMinLabel.textContent = state.anomalyMinSeverity.toFixed(1);
      if (!anomalySummary || !anomalyList) return;

      const minSev = _safeNum(state.anomalyMinSeverity, 2.0);
      const actorRows = (anomalyData.actorOrder || []).filter(row => _safeNum(row?.maxSeverity, 0.0) >= minSev);
      const issueEvents = (anomalyData.events || []).filter(ev => _safeNum(ev?.severity, 0.0) >= minSev);
      anomalySummary.textContent = actorRows.length > 0
        ? ('Flagged actors: ' + actorRows.length + '/' + anomalyData.totalTracks + ' | events>=' + minSev.toFixed(1) + ': ' + issueEvents.length)
        : ('No issues found above severity ' + minSev.toFixed(1));

      anomalyList.innerHTML = '';
      const activeIssue = (state.anomalyCursor >= 0 && state.anomalyCursor < issueEvents.length) ? issueEvents[state.anomalyCursor] : null;
      const activeIssueKey = activeIssue ? _eventKey(activeIssue) : '';
      const maxRows = 80;
      for (const row of actorRows.slice(0, maxRows)) {{
        const topEvent = row.events.find(ev => _safeNum(ev?.severity, 0.0) >= minSev) || row.events[0];
        const rowKey = topEvent ? _eventKey(topEvent) : '';
        const div = document.createElement('div');
        div.className = 'anomalyRow'
          + (activeIssueKey && rowKey && rowKey === activeIssueKey ? ' active' : '');
        const sevClass = _severityClass(topEvent ? topEvent.severity : row.maxSeverity);
        const topLabel = topEvent ? (String(topEvent.type) + '@' + _safeNum(topEvent.t, 0.0).toFixed(1) + 's') : '-';
        div.innerHTML = `<span class="anomalyDot ${{sevClass}}"></span>${{row.role}} #${{row.id}} (score=${{_safeNum(row.score, 0.0).toFixed(1)}}, events=${{row.eventCount}}, top=${{topLabel}})`;
        div.addEventListener('click', () => {{
          if (!topEvent) return;
          const filtered = (state.anomalyData?.events || []).filter(ev => _safeNum(ev?.severity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0));
          const key = _eventKey(topEvent);
          const idx = filtered.findIndex(ev => _eventKey(ev) === key);
          state.anomalyCursor = idx >= 0 ? idx : 0;
          const jumpEvent = idx >= 0 ? filtered[idx] : topEvent;
          const timelineIdx = _nearestTimelineIndex(_safeNum(jumpEvent.t, 0.0));
          state.tIndex = timelineIdx;
          slider.value = String(timelineIdx);
          state.focusActorId = String(jumpEvent.trackId);
          updateActorList();
          render();
        }});
        anomalyList.appendChild(div);
      }}
      if (actorRows.length > maxRows) {{
        const more = document.createElement('div');
        more.className = 'mono';
        more.style.color = '#dca9a9';
        more.textContent = '+' + String(actorRows.length - maxRows) + ' more actors';
        anomalyList.appendChild(more);
      }}
    }}

    function stepIssue(direction) {{
      const DATA = getCurrentData();
      state.anomalyData = computeScenarioAnomalies(DATA);
      const issueEvents = (state.anomalyData?.events || []).filter(ev => _safeNum(ev?.severity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0));
      if (!issueEvents.length) {{
        state.anomalyCursor = -1;
        updateAnomalyPanel();
        render();
        return;
      }}
      const dir = direction >= 0 ? 1 : -1;
      if (state.anomalyCursor < 0 || state.anomalyCursor >= issueEvents.length) {{
        state.anomalyCursor = dir > 0 ? 0 : (issueEvents.length - 1);
      }} else {{
        state.anomalyCursor = (state.anomalyCursor + dir + issueEvents.length) % issueEvents.length;
      }}
      const ev = issueEvents[state.anomalyCursor];
      const timelineIdx = _nearestTimelineIndex(_safeNum(ev?.t, 0.0));
      state.tIndex = timelineIdx;
      slider.value = String(timelineIdx);
      if (state.autoFocusIssue && ev) {{
        state.focusActorId = String(ev.trackId);
      }}
      updateActorList();
      render();
    }}

    function focusWorstIssueActor() {{
      const DATA = getCurrentData();
      state.anomalyData = computeScenarioAnomalies(DATA);
      const actorRows = (state.anomalyData?.actorOrder || []).filter(row => _safeNum(row?.maxSeverity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0));
      if (!actorRows.length) return;
      const top = actorRows[0];
      const ev = (top.events || []).find(e => _safeNum(e?.severity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0)) || top.events[0];
      state.focusActorId = String(top.id);
      if (ev) {{
        const issueEvents = (state.anomalyData?.events || []).filter(e => _safeNum(e?.severity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0));
        const idx = issueEvents.findIndex(e => _eventKey(e) === _eventKey(ev));
        state.anomalyCursor = idx >= 0 ? idx : -1;
        const timelineIdx = _nearestTimelineIndex(_safeNum(ev.t, 0.0));
        state.tIndex = timelineIdx;
        slider.value = String(timelineIdx);
      }}
      updateActorList();
      render();
    }}

    function getTimeline() {{
      const DATA = getCurrentData();
      return Array.isArray(DATA.timeline) ? DATA.timeline : [0.0];
    }}

    function resizeCanvas() {{
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
    }}

    function worldToScreen(x, y) {{
      const rect = canvas.getBoundingClientRect();
      return {{
        x: rect.width * 0.5 + (x - state.view.cx) * state.view.scale,
        y: rect.height * 0.5 - (y - state.view.cy) * state.view.scale,
      }};
    }}

    function screenToWorld(sx, sy) {{
      const rect = canvas.getBoundingClientRect();
      return {{
        x: state.view.cx + (sx - rect.width * 0.5) / state.view.scale,
        y: state.view.cy - (sy - rect.height * 0.5) / state.view.scale,
      }};
    }}

    function drawPolylineDirectionArrows(pts, color) {{
      if (!Array.isArray(pts) || pts.length < 2) return;
      if (!state.showCarlaDirection) return;
      if (_safeNum(state.view.scale, 1.0) < 0.45) return;

      const spacingPx = _safeNum(state.view.scale, 1.0) > 2.2 ? 48.0 : 64.0;
      const arrowLen = _safeNum(state.view.scale, 1.0) > 2.2 ? 8.0 : 6.5;
      const wing = arrowLen * 0.55;

      let travelled = 0.0;
      let nextMark = spacingPx;

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.3;
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.92;

      for (let i = 1; i < pts.length; i++) {{
        const p0 = worldToScreen(pts[i - 1][0], pts[i - 1][1]);
        const p1 = worldToScreen(pts[i][0], pts[i][1]);
        const dx = p1.x - p0.x;
        const dy = p1.y - p0.y;
        const segLen = Math.hypot(dx, dy);
        if (segLen < 1e-3) continue;

        while ((travelled + segLen) >= nextMark) {{
          const t = (nextMark - travelled) / segLen;
          const x = p0.x + t * dx;
          const y = p0.y + t * dy;
          const ux = dx / segLen;
          const uy = dy / segLen;
          const bx = x - ux * arrowLen;
          const by = y - uy * arrowLen;
          const nx = -uy;
          const ny = ux;

          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(bx + nx * wing, by + ny * wing);
          ctx.moveTo(x, y);
          ctx.lineTo(bx - nx * wing, by - ny * wing);
          ctx.stroke();

          nextMark += spacingPx;
        }}
        travelled += segLen;
      }}
      ctx.globalAlpha = 1.0;
    }}

    function fitView() {{
      const DATA = getCurrentData();
      const bbox = DATA.map_bbox;
      if (!bbox) return;
      const cx = (bbox.min_x + bbox.max_x) / 2;
      const cy = (bbox.min_y + bbox.max_y) / 2;
      const w = bbox.max_x - bbox.min_x;
      const h = bbox.max_y - bbox.min_y;
      const rect = canvas.getBoundingClientRect();
      const scale = Math.min(rect.width / (w * 1.1), rect.height / (h * 1.1));
      state.view.cx = cx;
      state.view.cy = cy;
      state.view.scale = Math.max(0.2, Math.min(500, scale));
      render();
    }}

    function findActorAtTime(track, t) {{
      const frames = track.frames || [];
      if (!frames.length) return null;
      let closest = frames[0];
      let minDiff = Math.abs(frames[0].t - t);
      for (const f of frames) {{
        const diff = Math.abs(f.t - t);
        if (diff < minDiff) {{
          minDiff = diff;
          closest = f;
        }}
      }}
      if (minDiff > 1.0) return null;
      return closest;
    }}

    function getRawPos(frame) {{
      if (!frame) return null;
      return {{ x: frame.x, y: frame.y, yaw: frame.yaw }};
    }}

    function getAlignedPos(frame) {{
      if (!frame) return null;
      if (frame.sx == null || frame.sy == null) return null;
      return {{
        x: frame.sx,
        y: frame.sy,
        yaw: frame.syaw != null ? frame.syaw : frame.yaw,
      }};
    }}

    function getCarlaPrePos(frame) {{
      if (!frame) return null;
      if (frame.cbx == null || frame.cby == null) return null;
      return {{
        x: frame.cbx,
        y: frame.cby,
        yaw: frame.cbyaw != null ? frame.cbyaw : (frame.syaw != null ? frame.syaw : frame.yaw),
      }};
    }}

    function getCarlaPos(frame) {{
      if (!frame) return null;
      if (frame.cx == null || frame.cy == null) return null;
      return {{
        x: frame.cx,
        y: frame.cy,
        yaw: frame.cyaw != null ? frame.cyaw : (frame.syaw != null ? frame.syaw : frame.yaw),
      }};
    }}

    function getDisplayPos(frame) {{
      if (state.showCarlaProjected) {{
        const carla = getCarlaPos(frame);
        if (carla) return carla;
      }}
      if (state.showCarlaPre) {{
        const carlaPre = getCarlaPrePos(frame);
        if (carlaPre) return carlaPre;
      }}
      if (state.showAligned) {{
        const aligned = getAlignedPos(frame);
        return aligned || getRawPos(frame);
      }}
      if (state.showRawLayer) {{
        return getRawPos(frame);
      }}
      // Fallback priority when raw layer is hidden.
      const aligned = getAlignedPos(frame);
      if (aligned) return aligned;
      const carlaPre = getCarlaPrePos(frame);
      if (carlaPre) return carlaPre;
      const carla = getCarlaPos(frame);
      if (carla) return carla;
      return getRawPos(frame);
    }}

    function render() {{
      const DATA = getCurrentData();
      const timeline = getTimeline();
      
      resizeCanvas();
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);

      const carlaImageRow = _ensureCarlaImageLoaded(DATA);
      if (state.showCarlaMap && state.showCarlaImage && carlaImageRow && carlaImageRow.ready && _isValidBounds(carlaImageRow.bounds)) {{
        const ib = carlaImageRow.bounds;
        const topLeft = worldToScreen(Number(ib.min_x), Number(ib.max_y));
        const bottomRight = worldToScreen(Number(ib.max_x), Number(ib.min_y));
        const x0 = Math.min(topLeft.x, bottomRight.x);
        const y0 = Math.min(topLeft.y, bottomRight.y);
        const w0 = Math.abs(bottomRight.x - topLeft.x);
        const h0 = Math.abs(bottomRight.y - topLeft.y);
        if (w0 > 1e-3 && h0 > 1e-3) {{
          ctx.globalAlpha = Math.max(0.0, Math.min(1.0, _safeNum(state.carlaImageOpacity, 0.55)));
          ctx.drawImage(carlaImageRow.img, x0, y0, w0, h0);
          ctx.globalAlpha = 1.0;
        }}
      }}

      // Draw lanes
      const lanes = DATA.lanes || [];
      for (const lane of lanes) {{
        const pts = lane.polyline || [];
        if (pts.length < 2) continue;
        ctx.beginPath();
        const p0 = worldToScreen(pts[0][0], pts[0][1]);
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i < pts.length; i++) {{
          const p = worldToScreen(pts[i][0], pts[i][1]);
          ctx.lineTo(p.x, p.y);
        }}
        ctx.strokeStyle = laneTypePalette[String(lane.lane_type)] || laneTypePalette['unknown'];
        ctx.lineWidth = 2;
        ctx.stroke();
      }}

      // Draw CARLA lane map (already transformed to V2 frame).
      if (state.showCarlaMap) {{
        const carlaMap = DATA.carla_map || null;
        const carlaLines = Array.isArray(carlaMap?.lines) ? carlaMap.lines : [];
        for (const line of carlaLines) {{
          const pts = line.polyline || [];
          if (pts.length < 2) continue;
          ctx.beginPath();
          const p0 = worldToScreen(pts[0][0], pts[0][1]);
          ctx.moveTo(p0.x, p0.y);
          for (let i = 1; i < pts.length; i++) {{
            const p = worldToScreen(pts[i][0], pts[i][1]);
            ctx.lineTo(p.x, p.y);
          }}
          const matchedCount = Number(line.matched_v2_count || 0);
          const laneColor = matchedCount > 0 ? 'rgba(59, 232, 255, 0.85)' : 'rgba(94, 123, 143, 0.45)';
          ctx.strokeStyle = laneColor;
          ctx.lineWidth = matchedCount > 0 ? 1.8 : 1.2;
          ctx.setLineDash(matchedCount > 0 ? [7, 4] : [3, 3]);
          ctx.stroke();
          ctx.setLineDash([]);
          const orientHint = String(line?.orientation_hint || '').toLowerCase();
          if (orientHint === 'forward') {{
            drawPolylineDirectionArrows(pts, laneColor);
          }} else if (orientHint === 'reversed') {{
            drawPolylineDirectionArrows([...pts].reverse(), laneColor);
          }}
        }}
        if (state.showLaneLabels) {{
          ctx.font = '10px monospace';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          ctx.fillStyle = '#d7ebf8';
          ctx.globalAlpha = 0.86;
          for (const line of carlaLines) {{
            const lx = Number(line?.label_x ?? line?.mid_x);
            const ly = Number(line?.label_y ?? line?.mid_y);
            if (!Number.isFinite(lx) || !Number.isFinite(ly)) continue;
            const s = worldToScreen(lx, ly);
            const lbl = String(line?.label || ('c' + String(line?.index ?? '?')));
            const matched = Array.isArray(line?.matched_v2_labels) ? line.matched_v2_labels : [];
            const txt = matched.length > 0 ? (lbl + ' <- ' + String(matched[0])) : lbl;
            const metrics = ctx.measureText(txt);
            ctx.fillStyle = 'rgba(0,0,0,0.66)';
            ctx.fillRect(s.x - 2, s.y - 2, metrics.width + 4, 12);
            ctx.fillStyle = '#d7ebf8';
            ctx.fillText(txt, s.x, s.y);
          }}
          ctx.globalAlpha = 1.0;
        }}
      }}
      
      // Draw lane labels if enabled
      if (state.showLaneLabels) {{
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (const lane of lanes) {{
          if (lane.mid_x == null || lane.mid_y == null) continue;
          const p = worldToScreen(lane.mid_x, lane.mid_y);
          const label = 'L' + (lane.lane_id || '?');
          // Draw background
          const metrics = ctx.measureText(label);
          ctx.fillStyle = 'rgba(0,0,0,0.7)';
          ctx.fillRect(p.x - metrics.width/2 - 2, p.y - 6, metrics.width + 4, 12);
          // Draw text
          ctx.fillStyle = '#fff';
          ctx.fillText(label, p.x, p.y);
        }}
      }}

      const t = timeline[state.tIndex] || 0;
      if (postShiftMinLabel) {{
        postShiftMinLabel.textContent = _safeNum(state.postShiftMinM, 1.5).toFixed(1) + 'm';
      }}
      const TRAJ_BREAK_DISTANCE = 8.0;
      const TRAJ_BREAK_TIME_GAP = 1.0;
      const tracks = DATA.tracks || [];
      state.anomalyData = state.anomalyData || computeScenarioAnomalies(DATA);
      const anomalyData = state.anomalyData || {{ byActor: {{}}, actorOrder: [], events: [], flaggedCount: 0, totalTracks: tracks.length }};
      const getTrackIssue = (track) => (anomalyData.byActor && typeof anomalyData.byActor === 'object')
        ? anomalyData.byActor[String(track.id)]
        : null;
      const hasTrackIssueAboveMin = (track) => {{
        const row = getTrackIssue(track);
        if (!row) return false;
        return _safeNum(row.maxSeverity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0);
      }};
      const hasFocusActor = String(state.focusActorId || '').length > 0;
      const isVisibleTrack = (track) => {{
        if (state.showEgoOnly && !isEgoTrack(track)) return false;
        if (state.showOnlyAnomalies && !hasTrackIssueAboveMin(track)) return false;
        return true;
      }};
      const isFocusedTrack = (track) => !hasFocusActor || (String(track.id) === String(state.focusActorId));
      const trackAlpha = (track) => {{
        if (!isVisibleTrack(track)) return 0.0;
        if (!hasFocusActor) return 1.0;
        if (isFocusedTrack(track)) return 1.0;
        return state.dimNonFocused ? 0.16 : 1.0;
      }};

      function drawTrackLayerPath(track, getPosFn, color, width, baseAlpha, dashPattern) {{
        const frames = track.frames || [];
        if (frames.length < 2) return;
        const alphaScale = trackAlpha(track);
        if (alphaScale <= 0.03) return;

        ctx.beginPath();
        let hasPath = false;
        for (let i = 0; i < frames.length; i++) {{
          const currFrame = frames[i];
          const currPos = getPosFn(currFrame);
          if (!currPos) continue;
          const currScreen = worldToScreen(currPos.x, currPos.y);

          if (i === 0) {{
            ctx.moveTo(currScreen.x, currScreen.y);
            hasPath = true;
            continue;
          }}

          const prevFrame = frames[i - 1];
          const prevPos = getPosFn(prevFrame);
          if (!prevPos) {{
            ctx.moveTo(currScreen.x, currScreen.y);
            hasPath = true;
            continue;
          }}

          const dx = currPos.x - prevPos.x;
          const dy = currPos.y - prevPos.y;
          const dist = Math.hypot(dx, dy);
          const dtFrame = (typeof currFrame.t === 'number' && typeof prevFrame.t === 'number')
            ? (currFrame.t - prevFrame.t)
            : 0.0;
          const breakSegment = (dtFrame < -1e-6) || (dtFrame > TRAJ_BREAK_TIME_GAP) || (dist > TRAJ_BREAK_DISTANCE);
          if (breakSegment) ctx.moveTo(currScreen.x, currScreen.y);
          else ctx.lineTo(currScreen.x, currScreen.y);
          hasPath = true;
        }}
        if (!hasPath) return;

        ctx.strokeStyle = color;
        ctx.globalAlpha = Math.max(0.03, Math.min(1.0, baseAlpha * alphaScale));
        ctx.lineWidth = isFocusedTrack(track) ? (width + 1.0) : width;
        ctx.setLineDash(Array.isArray(dashPattern) ? dashPattern : []);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
      }}

      function drawRawMarker(screen, radius, fillColor, alphaScale) {{
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = fillColor;
        ctx.globalAlpha = Math.max(0.1, alphaScale);
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = '#0b1822';
        ctx.lineWidth = 1;
        ctx.stroke();
      }}

    function drawAlignedMarker(screen, radius, alphaScale) {{
      const r = radius;
      ctx.beginPath();
      ctx.rect(screen.x - r, screen.y - r, 2 * r, 2 * r);
        ctx.fillStyle = '#ffffff';
        ctx.globalAlpha = Math.max(0.1, alphaScale);
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = '#ffb347';
      ctx.lineWidth = 2;
      ctx.stroke();
    }}

    function drawCarlaPreMarker(screen, radius, alphaScale) {{
      const r = radius;
      ctx.beginPath();
      ctx.moveTo(screen.x - r, screen.y + r);
      ctx.lineTo(screen.x + r, screen.y + r);
      ctx.lineTo(screen.x, screen.y - r);
      ctx.closePath();
      ctx.fillStyle = '#c78cff';
      ctx.globalAlpha = Math.max(0.1, alphaScale);
      ctx.fill();
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = '#f4d9ff';
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }}

    function drawCarlaMarker(screen, radius, alphaScale) {{
      const r = radius;
      ctx.beginPath();
      ctx.moveTo(screen.x, screen.y - r);
        ctx.lineTo(screen.x + r, screen.y);
        ctx.lineTo(screen.x, screen.y + r);
        ctx.lineTo(screen.x - r, screen.y);
        ctx.closePath();
        ctx.fillStyle = '#58ecff';
        ctx.globalAlpha = Math.max(0.1, alphaScale);
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = '#e8fdff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }}

      function drawEgoHalo(screen, alphaScale, focused) {{
        const outer = focused ? 20 : 16;
        const inner = focused ? 10 : 8;
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, outer, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(248, 198, 95, 0.18)';
        ctx.globalAlpha = Math.max(0.16, alphaScale * 0.55);
        ctx.fill();
        ctx.globalAlpha = 1.0;
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, inner, 0, Math.PI * 2);
        ctx.strokeStyle = focused ? '#ffe7a1' : '#f8c65f';
        ctx.globalAlpha = Math.max(0.24, alphaScale * 0.75);
        ctx.lineWidth = focused ? 2.8 : 2.0;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }}

      function drawMarkerTag(screen, tag, borderColor, alphaScale) {{
        if (!screen) return;
        const tx = screen.x + 6;
        const ty = screen.y - 10;
        ctx.font = 'bold 8px monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const metrics = ctx.measureText(tag);
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        ctx.globalAlpha = Math.max(0.15, alphaScale);
        ctx.fillRect(tx - 1, ty - 1, metrics.width + 4, 10);
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = '#ffffff';
        ctx.fillText(tag, tx + 1, ty + 1);
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(tx - 1, ty - 1, metrics.width + 4, 10);
      }}

      function drawAnomalyEventMarker(screen, severity, isCurrent, alphaScale, isFocused) {{
        if (!screen) return;
        const sev = Math.max(1.0, Math.min(5.0, _safeNum(severity, 1.0)));
        const radius = 3.0 + sev * 1.2 + (isCurrent ? 1.6 : 0.0);
        const color = sev >= 4.0 ? '#ff5f5f' : (sev >= 2.8 ? '#ff9f57' : '#f6d36f');
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.globalAlpha = isCurrent ? 0.95 : Math.max(0.2, alphaScale * (isFocused ? 0.9 : 0.58));
        ctx.lineWidth = isCurrent ? 2.4 : 1.3;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
        if (isCurrent) {{
          ctx.beginPath();
          ctx.moveTo(screen.x - radius - 2, screen.y);
          ctx.lineTo(screen.x + radius + 2, screen.y);
          ctx.moveTo(screen.x, screen.y - radius - 2);
          ctx.lineTo(screen.x, screen.y + radius + 2);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }}
      }}

      function drawPostShiftMarker(screen, shiftM, alphaScale) {{
        if (!screen) return;
        const s = Math.max(0.0, _safeNum(shiftM, 0.0));
        const radius = Math.max(2.5, Math.min(8.0, 2.0 + 0.9 * s));
        ctx.beginPath();
        ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 88, 88, 0.92)';
        ctx.globalAlpha = Math.max(0.2, Math.min(1.0, alphaScale));
        ctx.lineWidth = 1.3;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }}

      // Draw trajectories as explicit layers: raw vs V2 snapped vs CARLA projected.
      if (state.showTraj) {{
        for (const track of tracks) {{
          if (!isVisibleTrack(track)) continue;
          const roleColor = actorRolePalette[track.role] || actorRolePalette['npc'];
          const rawWidth = isEgoTrack(track) ? 2.4 : 1.4;
          const v2Width = isEgoTrack(track) ? 2.5 : 1.5;
          const carlaPreWidth = isEgoTrack(track) ? 2.6 : 1.6;
          const carlaWidth = isEgoTrack(track) ? 2.7 : 1.7;
          if (state.showRawLayer) {{
            drawTrackLayerPath(track, getRawPos, roleColor, rawWidth, 0.36, []);
          }}
          if (state.showAligned) {{
            drawTrackLayerPath(track, getAlignedPos, '#ffb347', v2Width, 0.8, [7, 5]);
          }}
          if (state.showCarlaPre) {{
            drawTrackLayerPath(track, getCarlaPrePos, '#c78cff', carlaPreWidth, 0.82, [6, 4]);
          }}
          if (state.showCarlaProjected) {{
            drawTrackLayerPath(track, getCarlaPos, '#58ecff', carlaWidth, 0.86, [2, 3]);
          }}
        }}
      }}

      if (state.showAnomalies) {{
        let anomalyDrawCount = 0;
        const drawLimit = 2800;
        for (const track of tracks) {{
          if (!isVisibleTrack(track)) continue;
          const issueRow = getTrackIssue(track);
          if (!issueRow || !Array.isArray(issueRow.events) || issueRow.events.length <= 0) continue;
          const frames = Array.isArray(track.frames) ? track.frames : [];
          const alphaScale = trackAlpha(track);
          if (alphaScale <= 0.03) continue;
          const focused = isFocusedTrack(track);
          for (const ev of issueRow.events) {{
            if (_safeNum(ev?.severity, 0.0) < _safeNum(state.anomalyMinSeverity, 2.0)) continue;
            const fi = Math.max(0, Math.min(frames.length - 1, Math.round(_safeNum(ev?.frameIndex, 0.0))));
            const fr = frames[fi];
            if (!fr) continue;
            const pos = getDisplayPos(fr) || getAlignedPos(fr) || getRawPos(fr);
            if (!pos) continue;
            const screen = worldToScreen(pos.x, pos.y);
            const isCurrent = Math.abs(_safeNum(fr.t, 0.0) - _safeNum(t, 0.0)) <= 0.11;
            drawAnomalyEventMarker(screen, _safeNum(ev.severity, 1.0), isCurrent, alphaScale, focused);
            anomalyDrawCount++;
            if (anomalyDrawCount >= drawLimit) break;
          }}
          if (anomalyDrawCount >= drawLimit) break;
        }}
      }}

      if (state.showPostShiftHotspots) {{
        const shiftMin = Math.max(0.0, _safeNum(state.postShiftMinM, 1.5));
        const drawLimit = 3200;
        let drawn = 0;
        for (const track of tracks) {{
          if (!isVisibleTrack(track)) continue;
          const alphaScale = trackAlpha(track);
          if (alphaScale <= 0.03) continue;
          const frames = Array.isArray(track.frames) ? track.frames : [];
          for (const fr of frames) {{
            if (!fr || fr.cbx == null || fr.cby == null || fr.cx == null || fr.cy == null) continue;
            const shift = Math.hypot(Number(fr.cx) - Number(fr.cbx), Number(fr.cy) - Number(fr.cby));
            if (!Number.isFinite(shift) || shift < shiftMin) continue;
            const sp = worldToScreen(Number(fr.cx), Number(fr.cy));
            drawPostShiftMarker(sp, shift, alphaScale);
            drawn++;
            if (drawn >= drawLimit) break;
          }}
          if (drawn >= drawLimit) break;
        }}
      }}

      // Draw current positions + layer deltas
      let activeCount = 0;
      let rawV2Count = 0;
      let rawV2Sum = 0.0;
      let rawCarlaPreCount = 0;
      let rawCarlaPreSum = 0.0;
      let rawCarlaCount = 0;
      let rawCarlaSum = 0.0;
      let carlaPreFinalCount = 0;
      let carlaPreFinalSum = 0.0;
      let v2CarlaCount = 0;
      let v2CarlaSum = 0.0;
      let activeEgoCount = 0;
      let activeIssueActorCount = 0;
      const totalEgoCount = tracks.filter(track => isEgoTrack(track)).length;
      const totalIssueActorCount = (anomalyData.actorOrder || []).filter(row => _safeNum(row?.maxSeverity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0)).length;
      for (const track of tracks) {{
        if (!isVisibleTrack(track)) continue;
        const frame = findActorAtTime(track, t);
        if (!frame) continue;
        activeCount++;
        if (isEgoTrack(track)) activeEgoCount++;
        if (hasTrackIssueAboveMin(track)) activeIssueActorCount++;

        const alphaScale = trackAlpha(track);
        if (alphaScale <= 0.03) continue;
        const color = actorRolePalette[track.role] || actorRolePalette['npc'];
        const rawPos = getRawPos(frame);
        const alignedPos = getAlignedPos(frame);
        const carlaPrePos = getCarlaPrePos(frame);
        const carlaPos = getCarlaPos(frame);
        const displayPos = getDisplayPos(frame);
        if (!displayPos) continue;
        const p = worldToScreen(displayPos.x, displayPos.y);
        const rawScreen = rawPos ? worldToScreen(rawPos.x, rawPos.y) : null;
        const alignedScreen = alignedPos ? worldToScreen(alignedPos.x, alignedPos.y) : null;
        const carlaPreScreen = carlaPrePos ? worldToScreen(carlaPrePos.x, carlaPrePos.y) : null;
        const carlaScreen = carlaPos ? worldToScreen(carlaPos.x, carlaPos.y) : null;
        const baseRadius = track.role === 'ego' ? 8 : 5;
        if (state.highlightEgo && isEgoTrack(track)) {{
          drawEgoHalo(p, alphaScale, isFocusedTrack(track));
        }}

        if (rawPos && alignedPos) {{
          rawV2Count++;
          rawV2Sum += Math.hypot(alignedPos.x - rawPos.x, alignedPos.y - rawPos.y);
        }}
        if (rawPos && carlaPrePos) {{
          rawCarlaPreCount++;
          rawCarlaPreSum += Math.hypot(carlaPrePos.x - rawPos.x, carlaPrePos.y - rawPos.y);
        }}
        if (rawPos && carlaPos) {{
          rawCarlaCount++;
          rawCarlaSum += Math.hypot(carlaPos.x - rawPos.x, carlaPos.y - rawPos.y);
        }}
        if (carlaPrePos && carlaPos) {{
          carlaPreFinalCount++;
          carlaPreFinalSum += Math.hypot(carlaPos.x - carlaPrePos.x, carlaPos.y - carlaPrePos.y);
        }}
        if (alignedPos && carlaPos) {{
          v2CarlaCount++;
          v2CarlaSum += Math.hypot(carlaPos.x - alignedPos.x, carlaPos.y - alignedPos.y);
        }}

        if (state.showBothDots) {{
          if (state.showDeltaVectors) {{
            if (state.showRawLayer && state.showAligned && rawScreen && alignedScreen) {{
              ctx.beginPath();
              ctx.moveTo(rawScreen.x, rawScreen.y);
              ctx.lineTo(alignedScreen.x, alignedScreen.y);
              ctx.strokeStyle = 'rgba(255, 186, 74, 0.72)';
              ctx.lineWidth = 1;
              ctx.setLineDash([4, 3]);
              ctx.stroke();
              ctx.setLineDash([]);
            }}
            if (state.showAligned && state.showCarlaPre && alignedScreen && carlaPreScreen) {{
              ctx.beginPath();
              ctx.moveTo(alignedScreen.x, alignedScreen.y);
              ctx.lineTo(carlaPreScreen.x, carlaPreScreen.y);
              ctx.strokeStyle = 'rgba(224, 172, 255, 0.72)';
              ctx.lineWidth = 1.1;
              ctx.setLineDash([3, 4]);
              ctx.stroke();
              ctx.setLineDash([]);
            }}
            if (state.showCarlaPre && state.showCarlaProjected && carlaPreScreen && carlaScreen) {{
              ctx.beginPath();
              ctx.moveTo(carlaPreScreen.x, carlaPreScreen.y);
              ctx.lineTo(carlaScreen.x, carlaScreen.y);
              ctx.strokeStyle = 'rgba(176, 108, 255, 0.80)';
              ctx.lineWidth = 1.2;
              ctx.setLineDash([2, 3]);
              ctx.stroke();
              ctx.setLineDash([]);
            }} else if (state.showAligned && state.showCarlaProjected && alignedScreen && carlaScreen) {{
              ctx.beginPath();
              ctx.moveTo(alignedScreen.x, alignedScreen.y);
              ctx.lineTo(carlaScreen.x, carlaScreen.y);
              ctx.strokeStyle = 'rgba(176, 108, 255, 0.78)';
              ctx.lineWidth = 1.2;
              ctx.setLineDash([3, 3]);
              ctx.stroke();
              ctx.setLineDash([]);
            }} else if (state.showRawLayer && state.showCarlaProjected && rawScreen && carlaScreen) {{
              ctx.beginPath();
              ctx.moveTo(rawScreen.x, rawScreen.y);
              ctx.lineTo(carlaScreen.x, carlaScreen.y);
              ctx.strokeStyle = 'rgba(88, 236, 255, 0.55)';
              ctx.lineWidth = 1;
              ctx.setLineDash([2, 3]);
              ctx.stroke();
              ctx.setLineDash([]);
            }}
          }}

          if (state.showRawLayer && rawScreen) {{
            drawRawMarker(rawScreen, Math.max(3, baseRadius - 2), color, alphaScale);
          }}
          if (state.showAligned && alignedScreen) {{
            drawAlignedMarker(alignedScreen, Math.max(4, baseRadius - 1), alphaScale);
          }}
          if (state.showCarlaPre && carlaPreScreen) {{
            drawCarlaPreMarker(carlaPreScreen, Math.max(4, baseRadius - 1), alphaScale);
          }}
          if (state.showCarlaProjected && carlaScreen) {{
            drawCarlaMarker(carlaScreen, baseRadius, alphaScale);
          }}
          if (isFocusedTrack(track)) {{
            if (state.showRawLayer && rawScreen) drawMarkerTag(rawScreen, 'R', '#6bc6ff', alphaScale);
            if (state.showAligned && alignedScreen) drawMarkerTag(alignedScreen, 'V', '#ffb347', alphaScale);
            if (state.showCarlaPre && carlaPreScreen) drawMarkerTag(carlaPreScreen, 'P', '#c78cff', alphaScale);
            if (state.showCarlaProjected && carlaScreen) drawMarkerTag(carlaScreen, 'C', '#58ecff', alphaScale);
          }}
        }} else {{
          ctx.beginPath();
          ctx.arc(p.x, p.y, baseRadius, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.globalAlpha = Math.max(0.1, alphaScale);
          ctx.fill();
          ctx.globalAlpha = 1.0;
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 1;
          ctx.stroke();
        }}
        
        // Draw direction arrow
        const yawRad = (displayPos.yaw || 0) * Math.PI / 180;
        const arrowLen = track.role === 'ego' ? 16 : 10;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + Math.cos(-yawRad + Math.PI/2) * arrowLen, p.y + Math.sin(-yawRad + Math.PI/2) * arrowLen);
        ctx.strokeStyle = color;
        ctx.globalAlpha = Math.max(0.1, alphaScale);
        ctx.lineWidth = isFocusedTrack(track) ? (isEgoTrack(track) ? 2.8 : 2.2) : (isEgoTrack(track) ? 2.2 : 1.6);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Draw actor ID label.
        if (state.showActorId && (!hasFocusActor || isFocusedTrack(track))) {{
          ctx.font = 'bold 9px monospace';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          const actorLabel = String(track.role) + ' #' + String(track.id);
          const actorMetrics = ctx.measureText(actorLabel);
          ctx.fillStyle = 'rgba(0,0,0,0.8)';
          ctx.fillRect(p.x + 10, p.y - 18, actorMetrics.width + 4, 12);
          ctx.fillStyle = '#fff';
          ctx.fillText(actorLabel, p.x + 12, p.y - 16);
        }}
        
        // Draw lane_id label if enabled and available
        if (state.showActorLane && (!hasFocusActor || isFocusedTrack(track)) && (frame.assigned_lane_id != null || frame.ccli != null || frame.cbcli != null)) {{
          ctx.font = 'bold 9px monospace';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          let laneLabel = 'L' + frame.assigned_lane_id + ' (r' + (frame.road_id || '?') + ')';
          if (state.showCarlaProjected) {{
            laneLabel = 'C' + ((frame.ccli != null) ? frame.ccli : '?') + ' [' + String(frame.csource || '-') + ']';
          }} else if (state.showCarlaPre) {{
            laneLabel = 'P' + ((frame.cbcli != null) ? frame.cbcli : '?');
          }}
          ctx.fillStyle = 'rgba(0,0,0,0.8)';
          const metrics = ctx.measureText(laneLabel);
          const laneLabelY = p.y - 4;
          ctx.fillRect(p.x + 10, laneLabelY, metrics.width + 4, 12);
          ctx.fillStyle = '#fff';
          ctx.fillText(laneLabel, p.x + 12, laneLabelY + 2);
        }}
      }}
      
      // Update alignment info
      const compareBits = [];
      if (rawV2Count > 0) compareBits.push('raw→V2 avg=' + (rawV2Sum / rawV2Count).toFixed(2) + 'm');
      if (rawCarlaPreCount > 0) compareBits.push('raw→CARLA(pre) avg=' + (rawCarlaPreSum / rawCarlaPreCount).toFixed(2) + 'm');
      if (rawCarlaCount > 0) compareBits.push('raw→CARLA avg=' + (rawCarlaSum / rawCarlaCount).toFixed(2) + 'm');
      if (carlaPreFinalCount > 0) compareBits.push('CARLA pre→final avg=' + (carlaPreFinalSum / carlaPreFinalCount).toFixed(2) + 'm');
      if (v2CarlaCount > 0) compareBits.push('V2→CARLA avg=' + (v2CarlaSum / v2CarlaCount).toFixed(2) + 'm');
      const focusTxt = hasFocusActor ? (' | focus=' + String(state.focusActorId)) : '';
      alignInfo.textContent = compareBits.length > 0
        ? ('Comparison: ' + compareBits.join(' | ') + focusTxt)
        : ('Comparison: no active aligned samples' + focusTxt);

      // Update HUD
      if (COMPARISON_MODE) {{
        const profileName = getProfileName(DATA) || '-';
        const baseScenarioName = String(
          (comparisonMeta && comparisonMeta.base_scenario)
          || DATA.base_scenario_name
          || DATA.scenario_name
          || '-'
        );
        const bestTag = (state.bestVariantProfile && profileName === state.bestVariantProfile) ? ' [BEST]' : '';
        hudScenario.textContent = 'Scenario: ' + baseScenarioName + ' | Variant: ' + profileName + bestTag;
      }} else {{
        hudScenario.textContent = 'Scenario: ' + (DATA.scenario_name || '-');
      }}
      const carlaMapName = (DATA.carla_map && DATA.carla_map.name) ? String(DATA.carla_map.name) : '';
      const modeParts = [];
      if (state.showRawLayer) modeParts.push('raw');
      if (state.showAligned) modeParts.push('v2');
      if (state.showCarlaPre) modeParts.push('carla_pre');
      if (state.showCarlaProjected) modeParts.push('carla');
      const modeTag = modeParts.length > 0 ? modeParts.join('+') : 'none';
      hudMap.textContent = 'Map: ' + (DATA.map_name || '-') + (carlaMapName ? (' | CARLA: ' + carlaMapName) : '') + ' | mode=' + modeTag;
      hudTime.textContent = 't=' + t.toFixed(3) + 's';
      const shownTrackCount = tracks.filter(track => isVisibleTrack(track)).length;
      const filterParts = [];
      if (state.showEgoOnly) filterParts.push('ego-only');
      if (state.showOnlyAnomalies) filterParts.push('anomaly-only');
      const modeSuffix = filterParts.length ? (' | filter=' + filterParts.join('+')) : '';
      hudCounts.textContent = 'Actors: ' + activeCount + '/' + shownTrackCount
        + ' | Egos: ' + activeEgoCount + '/' + totalEgoCount
        + ' | Issues: ' + activeIssueActorCount + '/' + totalIssueActorCount
        + (hasFocusActor ? (' | focus=' + String(state.focusActorId)) : '')
        + modeSuffix;
      timeLabel.textContent = 't=' + t.toFixed(3) + 's';
    }}

    function togglePlay() {{
      if (state.playing) {{
        clearInterval(state.playHandle);
        state.playing = false;
        playBtn.textContent = 'Play';
      }} else {{
        state.playing = true;
        playBtn.textContent = 'Pause';
        state.playHandle = setInterval(() => {{
          const timeline = getTimeline();
          state.tIndex = (state.tIndex + 1) % timeline.length;
          slider.value = String(state.tIndex);
          render();
        }}, 100);
      }}
    }}
    
    function updateActorList() {{
      const DATA = getCurrentData();
      state.anomalyData = computeScenarioAnomalies(DATA);
      const anomalyData = state.anomalyData || {{ byActor: {{}} }};
      actorList.innerHTML = '';
      const tracks = sortTracksForList(DATA.tracks || []);
      for (const track of tracks) {{
        if (state.showEgoOnly && !isEgoTrack(track)) continue;
        const actorId = String(track.id);
        const issueRow = (anomalyData.byActor && typeof anomalyData.byActor === 'object')
          ? anomalyData.byActor[actorId]
          : null;
        const hasIssue = !!(issueRow && _safeNum(issueRow.maxSeverity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0));
        if (state.showOnlyAnomalies && !hasIssue) continue;
        const div = document.createElement('div');
        div.className = 'actorRow'
          + (isEgoTrack(track) ? ' ego' : '')
          + ((state.focusActorId && actorId === String(state.focusActorId)) ? ' active' : '');
        const color = actorRolePalette[track.role] || actorRolePalette['npc'];
        const frames = Array.isArray(track.frames) ? track.frames : [];
        const cvals = frames
          .map(f => Number((f && typeof f === 'object') ? f.cdist : Number.NaN))
          .filter(v => Number.isFinite(v));
        let c90Txt = '-';
        if (cvals.length > 0) {{
          const sv = [...cvals].sort((a, b) => a - b);
          const p90 = sv[Math.min(sv.length - 1, Math.floor(0.9 * sv.length))];
          c90Txt = p90.toFixed(1) + 'm';
        }}
        const badgeHtml = isEgoTrack(track) ? '<span class="actorBadge ego">EGO</span>' : '';
        const issueBadgeHtml = hasIssue ? `<span class="actorBadge issue">ISSUE ${{_safeNum(issueRow.score, 0.0).toFixed(1)}}</span>` : '';
        const issueTxt = hasIssue ? (', issues=' + String(issueRow.eventCount)) : '';
        div.innerHTML = `<span class="actorDot" style="background:${{color}}"></span>${{badgeHtml}}${{issueBadgeHtml}}${{track.role}} #${{track.id}} (${{frames.length}} pts, c90=${{c90Txt}}${{issueTxt}})`;
        div.addEventListener('click', () => {{
          state.focusActorId = (String(state.focusActorId) === actorId) ? '' : actorId;
          const egoIds = getEgoActorIds(DATA.tracks || []);
          state.lastFocusedEgoIdx = egoIds.indexOf(String(state.focusActorId));
          updateActorList();
          render();
        }});
        actorList.appendChild(div);
      }}
      updateAnomalyPanel();
    }}
    
    function switchScenario(idx) {{
      currentScenarioIdx = idx;
      state.tIndex = 0;
      state.focusActorId = '';
      state.lastFocusedEgoIdx = -1;
      state.anomalyCursor = -1;
      state.anomalyData = computeScenarioAnomalies(getCurrentData());
      const timeline = getTimeline();
      slider.max = String(Math.max(0, timeline.length - 1));
      slider.value = '0';
      updateActorList();
      if (scenarioInfo) {{
        const DATA = getCurrentData();
        const tracks = DATA.tracks || [];
        const egoCount = tracks.filter(t => t.role === 'ego').length;
        const vehicleCount = tracks.filter(t => t.role === 'vehicle').length;
        const walkerCount = tracks.filter(t => t.role === 'walker').length;
        if (COMPARISON_MODE) {{
          const profileName = getProfileName(DATA);
          const bestTag = (state.bestVariantProfile && profileName === state.bestVariantProfile) ? ' [BEST]' : '';
          scenarioInfo.textContent = `Profile: ${{profileName || '-'}}${{bestTag}} | Ego: ${{egoCount}}, Vehicles: ${{vehicleCount}}, Walkers: ${{walkerCount}}, Frames: ${{timeline.length}}`;
        }} else {{
          scenarioInfo.textContent = `Ego: ${{egoCount}}, Vehicles: ${{vehicleCount}}, Walkers: ${{walkerCount}}, Frames: ${{timeline.length}}`;
        }}
      }}
      updateBestPickPanel();
      refreshScenarioSelectorLabels();
      _syncCarlaImageControls();
      fitView();
    }}

    // Build legend (static)
    for (const [type, color] of Object.entries(laneTypePalette)) {{
      const div = document.createElement('div');
      div.className = 'legendItem';
      div.innerHTML = `<span class="legendSwatch" style="background:${{color}}"></span>Lane type ${{type}}`;
      laneLegend.appendChild(div);
    }}
    const carlaLegendMapped = document.createElement('div');
    carlaLegendMapped.className = 'legendItem';
    carlaLegendMapped.innerHTML = `<span class="legendSwatch" style="background:#3be8ff"></span>CARLA lane (matched)`;
    laneLegend.appendChild(carlaLegendMapped);
    const carlaLegendUnmatched = document.createElement('div');
    carlaLegendUnmatched.className = 'legendItem';
    carlaLegendUnmatched.innerHTML = `<span class="legendSwatch" style="background:#5e7b8f"></span>CARLA lane (unmatched)`;
    laneLegend.appendChild(carlaLegendUnmatched);
    const rawTrailLegend = document.createElement('div');
    rawTrailLegend.className = 'legendItem';
    rawTrailLegend.innerHTML = `<span class="legendSwatch" style="background:#6bc6ff"></span>Raw actor trail`;
    laneLegend.appendChild(rawTrailLegend);
    const v2TrailLegend = document.createElement('div');
    v2TrailLegend.className = 'legendItem';
    v2TrailLegend.innerHTML = `<span class="legendSwatch" style="background:#ffb347"></span>V2 snapped trail`;
    laneLegend.appendChild(v2TrailLegend);
    const carlaPreTrailLegend = document.createElement('div');
    carlaPreTrailLegend.className = 'legendItem';
    carlaPreTrailLegend.innerHTML = `<span class="legendSwatch" style="background:#c78cff"></span>CARLA pre-postprocess trail`;
    laneLegend.appendChild(carlaPreTrailLegend);
    const carlaTrailLegend = document.createElement('div');
    carlaTrailLegend.className = 'legendItem';
    carlaTrailLegend.innerHTML = `<span class="legendSwatch" style="background:#58ecff"></span>CARLA final trail`;
    laneLegend.appendChild(carlaTrailLegend);

    for (const [role, color] of Object.entries(actorRolePalette)) {{
      const div = document.createElement('div');
      div.className = 'legendItem';
      div.innerHTML = `<span class="legendSwatch" style="background:${{color}}"></span>${{role}}`;
      actorLegend.appendChild(div);
    }}

    state.bestVariantProfile = loadBestVariantProfile();
    
    // Build scenario selector (multi mode)
    if (MULTI_MODE && scenarioSelect) {{
      for (let i = 0; i < scenarios.length; i++) {{
        const opt = document.createElement('option');
        opt.value = String(i);
        scenarioSelect.appendChild(opt);
      }}
      refreshScenarioSelectorLabels();
      scenarioSelect.addEventListener('change', () => {{
        switchScenario(parseInt(scenarioSelect.value, 10));
      }});
    }}

    if (COMPARISON_MODE && markBestBtn) {{
      markBestBtn.addEventListener('click', () => {{
        const profileName = getProfileName(getCurrentData());
        if (!profileName) return;
        saveBestVariantProfile(profileName);
        updateBestPickPanel();
        refreshScenarioSelectorLabels();
        render();
      }});
    }}
    if (COMPARISON_MODE && clearBestBtn) {{
      clearBestBtn.addEventListener('click', () => {{
        saveBestVariantProfile('');
        updateBestPickPanel();
        refreshScenarioSelectorLabels();
        render();
      }});
    }}

    // Events
    slider.addEventListener('input', () => {{
      state.tIndex = parseInt(slider.value, 10);
      render();
    }});
    playBtn.addEventListener('click', togglePlay);
    fitBtn.addEventListener('click', fitView);
    
    showRawLayerToggle.addEventListener('change', () => {{
      state.showRawLayer = showRawLayerToggle.checked;
      render();
    }});
    showAlignedToggle.addEventListener('change', () => {{
      state.showAligned = showAlignedToggle.checked;
      render();
    }});
    showCarlaPreToggle.addEventListener('change', () => {{
      state.showCarlaPre = showCarlaPreToggle.checked;
      render();
    }});
    showCarlaProjectedToggle.addEventListener('change', () => {{
      state.showCarlaProjected = showCarlaProjectedToggle.checked;
      render();
    }});
    showCarlaMapToggle.addEventListener('change', () => {{
      state.showCarlaMap = showCarlaMapToggle.checked;
      render();
    }});
    if (showCarlaDirectionToggle) {{
      showCarlaDirectionToggle.addEventListener('change', () => {{
        state.showCarlaDirection = showCarlaDirectionToggle.checked;
        render();
      }});
    }}
    if (showCarlaImageToggle) {{
      showCarlaImageToggle.addEventListener('change', () => {{
        state.showCarlaImage = showCarlaImageToggle.checked;
        render();
      }});
    }}
    if (carlaImageOpacitySlider) {{
      carlaImageOpacitySlider.addEventListener('input', () => {{
        const pct = _safeNum(carlaImageOpacitySlider.value, 55.0);
        state.carlaImageOpacity = Math.max(0.0, Math.min(1.0, pct / 100.0));
        if (carlaImageOpacityLabel) {{
          carlaImageOpacityLabel.textContent = String(Math.round(Math.max(0, Math.min(100, pct)))) + '%';
        }}
        render();
      }});
    }}
    showDeltaVectorsToggle.addEventListener('change', () => {{
      state.showDeltaVectors = showDeltaVectorsToggle.checked;
      render();
    }});
    showPostShiftHotspotsToggle.addEventListener('change', () => {{
      state.showPostShiftHotspots = showPostShiftHotspotsToggle.checked;
      render();
    }});
    postShiftMinSlider.addEventListener('input', () => {{
      const v = Math.max(0.0, Math.min(8.0, _safeNum(postShiftMinSlider.value, 15.0) / 10.0));
      state.postShiftMinM = v;
      if (postShiftMinLabel) postShiftMinLabel.textContent = v.toFixed(1) + 'm';
      render();
    }});
    showTrajToggle.addEventListener('change', () => {{
      state.showTraj = showTrajToggle.checked;
      render();
    }});
    showBothDotsToggle.addEventListener('change', () => {{
      state.showBothDots = showBothDotsToggle.checked;
      render();
    }});
    dimNonFocusedToggle.addEventListener('change', () => {{
      state.dimNonFocused = dimNonFocusedToggle.checked;
      render();
    }});
    showLaneLabelsToggle.addEventListener('change', () => {{
      state.showLaneLabels = showLaneLabelsToggle.checked;
      render();
    }});
    showActorIdToggle.addEventListener('change', () => {{
      state.showActorId = showActorIdToggle.checked;
      render();
    }});
    showActorLaneToggle.addEventListener('change', () => {{
      state.showActorLane = showActorLaneToggle.checked;
      render();
    }});
    showAnomaliesToggle.addEventListener('change', () => {{
      state.showAnomalies = showAnomaliesToggle.checked;
      render();
    }});
    showOnlyAnomaliesToggle.addEventListener('change', () => {{
      state.showOnlyAnomalies = showOnlyAnomaliesToggle.checked;
      if (state.showOnlyAnomalies && state.focusActorId) {{
        const DATA = getCurrentData();
        state.anomalyData = computeScenarioAnomalies(DATA);
        const issueRow = state.anomalyData?.byActor?.[String(state.focusActorId)] || null;
        const hasIssue = !!(issueRow && _safeNum(issueRow.maxSeverity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0));
        if (!hasIssue) state.focusActorId = '';
      }}
      updateActorList();
      render();
    }});
    autoFocusIssueToggle.addEventListener('change', () => {{
      state.autoFocusIssue = autoFocusIssueToggle.checked;
      updateAnomalyPanel();
      render();
    }});
    anomalyMinSlider.addEventListener('input', () => {{
      state.anomalyMinSeverity = _safeNum(anomalyMinSlider.value, 2.0);
      if (state.showOnlyAnomalies && state.focusActorId) {{
        const DATA = getCurrentData();
        state.anomalyData = computeScenarioAnomalies(DATA);
        const issueRow = state.anomalyData?.byActor?.[String(state.focusActorId)] || null;
        const hasIssue = !!(issueRow && _safeNum(issueRow.maxSeverity, 0.0) >= _safeNum(state.anomalyMinSeverity, 2.0));
        if (!hasIssue) state.focusActorId = '';
      }}
      state.anomalyCursor = -1;
      updateActorList();
      render();
    }});
    highlightEgoToggle.addEventListener('change', () => {{
      state.highlightEgo = highlightEgoToggle.checked;
      render();
    }});
    showEgoOnlyToggle.addEventListener('change', () => {{
      state.showEgoOnly = showEgoOnlyToggle.checked;
      if (state.showEgoOnly && state.focusActorId) {{
        const DATA = getCurrentData();
        const focusedTrack = (DATA.tracks || []).find(track => String(track.id) === String(state.focusActorId));
        if (focusedTrack && !isEgoTrack(focusedTrack)) {{
          state.focusActorId = '';
        }}
      }}
      updateActorList();
      render();
    }});
    prevIssueBtn.addEventListener('click', () => stepIssue(-1));
    nextIssueBtn.addEventListener('click', () => stepIssue(1));
    focusWorstIssueBtn.addEventListener('click', focusWorstIssueActor);
    clearFocusBtn.addEventListener('click', () => {{
      state.focusActorId = '';
      state.lastFocusedEgoIdx = -1;
      updateActorList();
      render();
    }});
    focusEgoBtn.addEventListener('click', focusFirstEgo);
    cycleEgoBtn.addEventListener('click', cycleEgoFocus);

    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      state.view.scale = Math.max(0.1, Math.min(500, state.view.scale * factor));
      render();
    }});

    canvas.addEventListener('mousedown', (e) => {{
      state.drag = {{ x: e.clientX, y: e.clientY, cx: state.view.cx, cy: state.view.cy }};
    }});
    canvas.addEventListener('mousemove', (e) => {{
      if (!state.drag) return;
      const dx = e.clientX - state.drag.x;
      const dy = e.clientY - state.drag.y;
      state.view.cx = state.drag.cx - dx / state.view.scale;
      state.view.cy = state.drag.cy + dy / state.view.scale;
      render();
    }});
    canvas.addEventListener('mouseup', () => {{ state.drag = null; }});
    canvas.addEventListener('mouseleave', () => {{ state.drag = null; }});

    window.addEventListener('resize', render);
    
    // Initial setup
    switchScenario(0);
  }})();
  </script>
</body>
</html>"""


def build_dataset(
    scenario_name: str,
    chosen_map: VectorMapData,
    ego_trajs: List[List[Waypoint]],
    ego_times: List[List[float]],
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    obj_info: Dict[int, Dict[str, object]],
    dt: float,
    matcher: Optional[LaneMatcher] = None,
    carla_context: Optional[Dict[str, object]] = None,
    timing_optimization: Optional[Dict[str, object]] = None,
    walker_processing: Optional[Dict[str, object]] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """Build the dataset dictionary for HTML visualization."""
    t_build_dataset_start = time.perf_counter()

    # Create trajectory aligners if matcher available
    ego_aligner: Optional[TrajectoryAligner] = None
    vehicle_aligner: Optional[TrajectoryAligner] = None
    vehicle_aligner_low_motion: Optional[TrajectoryAligner] = None
    if matcher is not None:
        ego_aligner = TrajectoryAligner(matcher, verbose=verbose)
        # User requirement: vehicles snap only to lane type 1.
        vehicle_aligner = TrajectoryAligner(
            matcher,
            verbose=verbose,
            allowed_lane_types={"1"},
        )
        # For low-motion/parked-like vehicles, allow shoulder/parking-adjacent
        # lane type 3 to avoid dropping to raw fallback in corridor curb lanes.
        vehicle_aligner_low_motion = TrajectoryAligner(
            matcher,
            verbose=verbose,
            allowed_lane_types={"1", "3"},
        )

    # Guardrail: do not force extreme map snaps into aligned layer. Extremely far
    # snaps are usually mismatch artifacts and are safer to keep at raw pose.
    EGO_MAX_SNAP_DIST_M = _env_float("V2X_EGO_MAX_SNAP_DIST_M", 8.0)
    VEHICLE_MAX_SNAP_DIST_M = _env_float("V2X_VEHICLE_MAX_SNAP_DIST_M", 8.0)
    VEHICLE_ADAPTIVE_SNAP_REJECT = _env_int("V2X_VEHICLE_ADAPTIVE_SNAP_REJECT", 1, minimum=0, maximum=1) == 1
    VEHICLE_ADAPTIVE_MAX_SNAP_DIST_M = _env_float("V2X_VEHICLE_ADAPTIVE_MAX_SNAP_DIST_M", 2.6)
    VEHICLE_ADAPTIVE_SNAP_STEP_SCALE = _env_float("V2X_VEHICLE_ADAPTIVE_SNAP_STEP_SCALE", 2.2)
    VEHICLE_ADAPTIVE_SNAP_STEP_BIAS_M = _env_float("V2X_VEHICLE_ADAPTIVE_SNAP_STEP_BIAS_M", 0.8)
    VEHICLE_ADAPTIVE_SNAP_LOW_STEP_M = _env_float("V2X_VEHICLE_ADAPTIVE_SNAP_LOW_STEP_M", 0.8)
    
    # Build timeline
    all_times: List[float] = []
    for times in ego_times:
        all_times.extend(times)
    for times in vehicle_times.values():
        all_times.extend(times)
    if all_times:
        timeline = sorted(set(all_times))
    else:
        # Fallback: use frame indices
        max_frames = max(
            (len(t) for t in ego_trajs),
            default=0,
        )
        max_frames = max(
            max_frames,
            max((len(t) for t in vehicles.values()), default=0),
        )
        timeline = [float(i) * dt for i in range(max_frames)]

    lane_to_carla_meta: Dict[int, Dict[str, object]] = {}
    if isinstance(carla_context, dict):
        lane_to_carla_raw = carla_context.get("lane_to_carla", {})
        if isinstance(lane_to_carla_raw, dict):
            for li, row in lane_to_carla_raw.items():
                li_i = _safe_int(li, -1)
                if li_i >= 0 and isinstance(row, dict):
                    lane_to_carla_meta[int(li_i)] = dict(row)

    # Build lanes data with road_id and lane_id for visualization
    lanes_data: List[Dict[str, object]] = []
    for lane in chosen_map.lanes:
        # Compute lane midpoint for label placement
        poly = lane.polyline
        mid_idx = len(poly) // 2
        mid_pt = poly[mid_idx] if len(poly) > 0 else [0, 0, 0]
        lane_row: Dict[str, object] = {
            "uid": lane.uid,
            "index": int(lane.index),
            "lane_type": lane.lane_type,
            "road_id": lane.road_id,
            "lane_id": lane.lane_id,
            "polyline": poly[:, :2].tolist(),
            "mid_x": float(mid_pt[0]),
            "mid_y": float(mid_pt[1]),
        }
        cm = lane_to_carla_meta.get(int(lane.index))
        if isinstance(cm, dict):
            split_extras = cm.get("split_extra_carla_lines", [])
            lane_row["carla_match"] = {
                "carla_line_index": int(_safe_int(cm.get("carla_line_index"), -1)),
                "carla_line_label": f"c{int(_safe_int(cm.get('carla_line_index'), -1))}",
                "reversed": bool(cm.get("reversed", False)),
                "quality": str(cm.get("quality", "poor")),
                "usable": bool(str(cm.get("quality", "poor")) != "poor"),
                "score": float(_safe_float(cm.get("score"), float("inf"))),
                "median_dist_m": float(_safe_float(cm.get("median_dist_m"), float("inf"))),
                "p90_dist_m": float(_safe_float(cm.get("p90_dist_m"), float("inf"))),
                "coverage_2m": float(_safe_float(cm.get("coverage_2m"), 0.0)),
                "angle_median_deg": float(_safe_float(cm.get("angle_median_deg"), 180.0)),
                "monotonic_ratio": float(_safe_float(cm.get("monotonic_ratio"), 0.0)),
                "length_ratio": float(_safe_float(cm.get("length_ratio"), 0.0)),
                "shared_carla_line": bool(cm.get("shared_carla_line", False)),
                "split_extra_carla_lines": [
                    int(_safe_int(v, -1))
                    for v in (split_extras if isinstance(split_extras, (list, tuple, set)) else [])
                    if _safe_int(v, -1) >= 0
                ],
            }
        else:
            lane_row["carla_match"] = None
        lanes_data.append(lane_row)

    # Build tracks
    tracks: List[Dict[str, object]] = []

    def _dedupe_frames_by_time(
        frames: List[Dict[str, object]],
        dt_hint: float,
    ) -> List[Dict[str, object]]:
        """
        Remove duplicate frames that share the same scenario time bucket.
        This prevents merged cross-ID branches from creating repeated loops that
        later appear as random lane-change-and-return artifacts.
        """
        if len(frames) <= 1:
            return frames

        step = float(dt_hint) if (isinstance(dt_hint, (int, float)) and dt_hint > 1e-6) else 0.1
        inv_step = 1.0 / float(step)
        best_by_key: Dict[int, Tuple[int, float]] = {}

        def _frame_rank(fr: Dict[str, object]) -> float:
            # Lower is better: prefer valid aligned lane snaps with smaller snap distance.
            lane_idx = _safe_int(fr.get("lane_index"), -1)
            if lane_idx < 0:
                return 1e6
            sdist = _safe_float(fr.get("sdist"), float("inf"))
            if not math.isfinite(sdist):
                sdist = 1e5
            # Mildly prefer synthetic turn anchors when score ties.
            syn = 0.0 if bool(fr.get("synthetic_turn", False)) else 0.05
            return float(sdist) + float(syn)

        for i, fr in enumerate(frames):
            t = _safe_float(fr.get("t"), float(i) * float(step))
            key = int(round(float(t) * inv_step))
            rank = _frame_rank(fr)
            prev = best_by_key.get(key)
            if prev is None or rank < float(prev[1]):
                best_by_key[key] = (int(i), float(rank))

        # Emit strictly chronological order by time-bucket key.
        keep_keys = sorted(best_by_key.keys())
        out = [frames[int(best_by_key[k][0])] for k in keep_keys]
        return out
    
    # Ego tracks - use trajectory-level alignment
    for ego_idx, (traj, times) in enumerate(zip(ego_trajs, ego_times)):
        # Get trajectory-level alignment
        snap_results: List[Optional[Dict[str, object]]] = []
        if ego_aligner is not None and len(traj) > 0:
            if verbose:
                print(f"  [BUILD] Aligning ego_{ego_idx} ({len(traj)} frames)")
            snap_results = ego_aligner.align_trajectory(traj)
        
        frames = []
        for i, wp in enumerate(traj):
            t = times[i] if i < len(times) else float(i) * dt
            frame: Dict[str, object] = {
                "t": t,
                "x": float(wp.x),
                "y": float(wp.y),
                "z": float(wp.z),
                "yaw": float(wp.yaw),
            }
            # Apply trajectory-aligned snap
            if i < len(snap_results) and snap_results[i] is not None:
                snap = snap_results[i]
                snap_dist = float(snap.get("sdist", snap.get("dist", 0.0)))
                if math.isfinite(snap_dist) and snap_dist > float(EGO_MAX_SNAP_DIST_M):
                    frame["sx"] = float(wp.x)
                    frame["sy"] = float(wp.y)
                    frame["sz"] = float(wp.z)
                    frame["syaw"] = float(wp.yaw)
                    frame["sdist"] = 0.0
                    frame["snap_rejected_far"] = True
                    frame["snap_reject_dist"] = float(snap_dist)
                    frame["lane_index"] = -1
                    frame["road_id"] = -1
                    frame["lane_id"] = 0
                    frame["assigned_lane_id"] = 0
                else:
                    frame["sx"] = float(snap["x"])
                    frame["sy"] = float(snap["y"])
                    frame["sz"] = float(snap["z"])
                    frame["syaw"] = float(snap["yaw"])
                    frame["sdist"] = float(snap_dist)
                    frame["slane"] = str(snap.get("lane_type", ""))
                    frame["lane_index"] = int(snap.get("lane_index", -1))
                    frame["road_id"] = int(snap.get("road_id", -1))
                    frame["lane_id"] = int(snap.get("lane_id", 0))
                    frame["assigned_lane_id"] = int(snap.get("assigned_lane_id", snap.get("lane_id", 0)))
                    if bool(snap.get("synthetic_turn", False)):
                        frame["synthetic_turn"] = True
                        frame["turn_u"] = float(snap.get("turn_u", 0.0))
            frames.append(frame)
        frames = _dedupe_frames_by_time(frames, dt_hint=dt)
        tracks.append({
            "id": f"ego_{ego_idx}",
            "role": "ego",
            "obj_type": "ego",
            "length": 4.8,
            "width": 2.1,
            "frames": frames,
        })

    # Vehicle tracks - use trajectory-level alignment
    for vid, traj in vehicles.items():
        times = vehicle_times.get(vid, [])
        meta = obj_info.get(vid, {})
        obj_type = str(meta.get("obj_type", "npc"))
        
        # Determine role
        obj_lower = obj_type.lower()
        if "pedestrian" in obj_lower or "walker" in obj_lower or "child" in obj_lower:
            role = "walker"
        elif "cyclist" in obj_lower or "bicycle" in obj_lower or "scooter" in obj_lower:
            role = "cyclist"
        else:
            role = "vehicle"

        # Get trajectory-level alignment (only for vehicles)
        snap_results: List[Optional[Dict[str, object]]] = []
        low_motion_vehicle = False
        if vehicle_aligner is not None and role == "vehicle" and len(traj) > 0:
            path_len = float(_trajectory_path_length_m(traj))
            if len(traj) >= 2:
                net_disp = float(math.hypot(float(traj[-1].x) - float(traj[0].x), float(traj[-1].y) - float(traj[0].y)))
            else:
                net_disp = 0.0
            if times and len(times) >= 2:
                duration_s = max(0.1, float(times[-1]) - float(times[0]))
            else:
                duration_s = max(0.1, float(max(1, len(traj) - 1)) * float(dt))
            avg_speed_mps = float(path_len / max(0.1, duration_s))

            low_motion_path_m = _env_float("V2X_LOW_MOTION_PATH_M", 12.0)
            low_motion_net_m = _env_float("V2X_LOW_MOTION_NET_M", 5.0)
            low_motion_speed_mps = _env_float("V2X_LOW_MOTION_AVG_SPEED_MPS", 1.6)
            low_motion_vehicle = (
                path_len <= float(low_motion_path_m)
                and net_disp <= float(low_motion_net_m)
                and avg_speed_mps <= float(low_motion_speed_mps)
            )
            aligner = (
                vehicle_aligner_low_motion
                if low_motion_vehicle and vehicle_aligner_low_motion is not None
                else vehicle_aligner
            )
            snap_results = aligner.align_trajectory(traj)

        frames = []
        for i, wp in enumerate(traj):
            t = times[i] if i < len(times) else float(i) * dt
            frame: Dict[str, object] = {
                "t": t,
                "x": float(wp.x),
                "y": float(wp.y),
                "z": float(wp.z),
                "yaw": float(wp.yaw),
            }
            # Apply trajectory-aligned snap (only for vehicles)
            if i < len(snap_results) and snap_results[i] is not None:
                snap = snap_results[i]
                snap_dist = float(snap.get("sdist", snap.get("dist", 0.0)))
                reject_snap = bool(math.isfinite(snap_dist) and snap_dist > float(VEHICLE_MAX_SNAP_DIST_M))
                reject_reason = "far"
                adaptive_raw_step = float("nan")
                adaptive_limit = float("nan")
                if (
                    (not bool(reject_snap))
                    and bool(VEHICLE_ADAPTIVE_SNAP_REJECT)
                    and int(i) > 0
                    and int(i) < len(traj)
                ):
                    raw_step = float(
                        math.hypot(
                            float(wp.x) - float(traj[i - 1].x),
                            float(wp.y) - float(traj[i - 1].y),
                        )
                    )
                    adaptive_raw_step = float(raw_step)
                    adaptive_limit = max(
                        float(VEHICLE_ADAPTIVE_MAX_SNAP_DIST_M),
                        float(VEHICLE_ADAPTIVE_SNAP_STEP_SCALE) * float(raw_step) + float(VEHICLE_ADAPTIVE_SNAP_STEP_BIAS_M),
                    )
                    if float(raw_step) <= float(VEHICLE_ADAPTIVE_SNAP_LOW_STEP_M) and float(snap_dist) > float(adaptive_limit):
                        reject_snap = True
                        reject_reason = "adaptive_low_motion_far"
                if bool(reject_snap):
                    frame["sx"] = float(wp.x)
                    frame["sy"] = float(wp.y)
                    frame["sz"] = float(wp.z)
                    frame["syaw"] = float(wp.yaw)
                    frame["sdist"] = 0.0
                    frame["snap_rejected_far"] = True
                    frame["snap_reject_dist"] = float(snap_dist)
                    frame["snap_reject_reason"] = str(reject_reason)
                    if math.isfinite(float(adaptive_raw_step)):
                        frame["snap_reject_raw_step"] = float(adaptive_raw_step)
                    if math.isfinite(float(adaptive_limit)):
                        frame["snap_reject_limit"] = float(adaptive_limit)
                    frame["lane_index"] = -1
                    frame["road_id"] = -1
                    frame["lane_id"] = 0
                    frame["assigned_lane_id"] = 0
                else:
                    frame["sx"] = float(snap["x"])
                    frame["sy"] = float(snap["y"])
                    frame["sz"] = float(snap["z"])
                    frame["syaw"] = float(snap["yaw"])
                    frame["sdist"] = float(snap_dist)
                    frame["slane"] = str(snap.get("lane_type", ""))
                    frame["lane_index"] = int(snap.get("lane_index", -1))
                    frame["road_id"] = int(snap.get("road_id", -1))
                    frame["lane_id"] = int(snap.get("lane_id", 0))
                    frame["assigned_lane_id"] = int(snap.get("assigned_lane_id", snap.get("lane_id", 0)))
                    if bool(snap.get("synthetic_turn", False)):
                        frame["synthetic_turn"] = True
                        frame["turn_u"] = float(snap.get("turn_u", 0.0))
            frames.append(frame)
        frames = _dedupe_frames_by_time(frames, dt_hint=dt)
        # Assign a deterministic blueprint per actor based on (vid, obj_type).
        # Using a seeded RNG ensures the same actor always gets the same model
        # across pipeline re-runs, while still producing variety across actors.
        import random as _random_mod
        from v2xpnp.pipeline.trajectory_ingest_stage_01_types_io import (
            map_obj_type as _map_obj_type,
        )
        _actor_rng = _random_mod.Random(hash((int(vid), str(obj_type))) & 0xFFFFFFFF)
        actor_model = str(meta.get("model") or "").strip() or _map_obj_type(obj_type, rng=_actor_rng)

        tr_entry: Dict[str, object] = {
            "id": int(vid),
            "role": role,
            "obj_type": obj_type,
            "model": actor_model,
            "low_motion_vehicle": bool(low_motion_vehicle),
            "frames": frames,
        }
        meta_len = _safe_float(meta.get("length"), float("nan"))
        meta_wid = _safe_float(meta.get("width"), float("nan"))
        if math.isfinite(meta_len) and 1.6 <= float(meta_len) <= 28.0:
            tr_entry["length"] = float(meta_len)
        if math.isfinite(meta_wid) and 0.8 <= float(meta_wid) <= 4.5:
            tr_entry["width"] = float(meta_wid)
        tracks.append(tr_entry)

    # Optional CARLA projection layer / trajectories
    carla_overlap_postprocess: Optional[Dict[str, object]] = None
    if carla_context is not None and bool(carla_context.get("enabled", False)):
        print(
            "[PERF] build_dataset CARLA stage start: "
            f"scenario={scenario_name} tracks={len(tracks)}"
        )
        t_apply_projection = time.perf_counter()
        apply_carla_projection_to_tracks(tracks, carla_context=carla_context)
        print(
            "[PERF] build_dataset CARLA stage apply_carla_projection_to_tracks: "
            f"elapsed={time.perf_counter() - t_apply_projection:.2f}s"
        )
        t_overlap = time.perf_counter()
        carla_overlap_postprocess = _reduce_carla_overlap_with_parked_guidance(
            tracks=tracks,
            carla_context=carla_context,
            scenario_name=str(scenario_name),
            verbose=bool(verbose),
        )
        print(
            "[PERF] build_dataset CARLA stage overlap_postprocess: "
            f"elapsed={time.perf_counter() - t_overlap:.2f}s"
        )
        # Pre-pass: merge duplicate-detection tracks (same physical vehicle
        # detected by multiple sensor agents and never deduped). This must
        # run before the residual collision resolver because the resolver
        # operates frame-by-frame and cannot detect long-term pairwise duplication.
        t_dup = time.perf_counter()
        duplicate_merge_report = _merge_duplicate_vehicle_tracks(
            tracks=tracks,
            scenario_name=str(scenario_name),
            verbose=bool(verbose),
        )
        if isinstance(duplicate_merge_report, dict):
            n_pairs = len(duplicate_merge_report.get("duplicate_pairs", []) or [])
            n_dropped = len(duplicate_merge_report.get("tracks_dropped", []) or [])
            if n_pairs or n_dropped:
                print(
                    f"[INFO] Duplicate-track merger: {n_pairs} pair(s), "
                    f"{n_dropped} track(s) dropped: "
                    f"{duplicate_merge_report.get('tracks_dropped')}  "
                    f"(elapsed {time.perf_counter() - t_dup:.2f}s)"
                )

        # Final-pass residual-collision resolver: any sustained pair overlap
        # that survived the upstream reducer gets fixed by un-snapping toward
        # raw (cx, cy ← x, y), or by deleting the less significant track.
        t_residual = time.perf_counter()
        residual_collision_report = _resolve_residual_vehicle_collisions(
            tracks=tracks,
            scenario_name=str(scenario_name),
            verbose=bool(verbose),
        )
        if isinstance(residual_collision_report, dict):
            n_runs = int(residual_collision_report.get("collision_runs", 0))
            n_unsnap = int(residual_collision_report.get("frames_unsnapped", 0))
            n_deleted = len(residual_collision_report.get("tracks_deleted_ids", []) or [])
            if n_runs or n_unsnap or n_deleted:
                print(
                    f"[INFO] Residual collision resolver: {n_runs} run(s), "
                    f"{n_unsnap} frame(s) un-snapped, {n_deleted} track(s) deleted "
                    f"(elapsed {time.perf_counter() - t_residual:.2f}s)"
                )
        # Final safety after overlap postprocess: some overlap repairs can
        # re-introduce low-motion teleport jumps.
        t_final_safety = time.perf_counter()
        final_safety_tracks = 0
        for tr in tracks:
            if not isinstance(tr, dict):
                continue
            role = str(tr.get("role", "")).strip().lower()
            if role not in {"ego", "vehicle"}:
                continue
            frames = tr.get("frames", [])
            if isinstance(frames, list) and frames:
                final_safety_tracks += 1
                # Parked invariant placement is a terminal pose lock; avoid
                # re-introducing wrong-way/fallback edits afterward.
                if any(str(fr.get("csource", "")).startswith("parked_invariant_") for fr in frames if isinstance(fr, dict)):
                    continue
                _suppress_low_motion_teleports(
                    frames,
                    raw_step_threshold_m=_env_float("V2X_CARLA_FINAL_TELEPORT_RAW_STEP_M", 0.6),
                    jump_threshold_m=_env_float("V2X_CARLA_FINAL_TELEPORT_JUMP_M", 1.8),
                    target_floor_m=_env_float("V2X_CARLA_FINAL_TELEPORT_TARGET_FLOOR_M", 1.2),
                    target_scale=_env_float("V2X_CARLA_FINAL_TELEPORT_TARGET_SCALE", 2.0),
                    target_bias_m=_env_float("V2X_CARLA_FINAL_TELEPORT_TARGET_BIAS_M", 0.4),
                )
                _fix_tail_low_motion_line_spike(
                    frames,
                    raw_step_threshold_m=_env_float("V2X_CARLA_FINAL_TAIL_LINE_HOLD_RAW_STEP_M", 0.6),
                    max_tail_run_frames=_env_int("V2X_CARLA_FINAL_TAIL_LINE_HOLD_MAX_RUN", 2, minimum=1, maximum=6),
                    min_prev_run_frames=_env_int("V2X_CARLA_FINAL_TAIL_LINE_HOLD_MIN_PREV_RUN", 2, minimum=1, maximum=12),
                )
                _fallback_wrong_way_carla_samples(
                    frames,
                    opposite_reject_deg=_env_float("V2X_CARLA_WRONG_WAY_REJECT_DEG", 178.0),
                    carla_context=carla_context,
                    canonical_reject_deg=_env_float("V2X_CARLA_WRONG_WAY_CANONICAL_REJECT_DEG", 181.0),
                    try_reproject=False,
                )
                _commit_intersection_connectors(
                    frames,
                    carla_context=carla_context,
                )
                # Clean up isolated fallback artifacts that can appear after
                # wrong-way rejection on otherwise stable lane-following runs.
                for fi in range(1, max(1, len(frames) - 1)):
                    fr = frames[fi]
                    src = str(fr.get("csource", "")).strip().lower()
                    if src not in {"raw_fallback_wrong_way", "raw_fallback_far", "raw_fallback"}:
                        continue
                    prev_fr = frames[fi - 1]
                    next_fr = frames[fi + 1]
                    if not (_frame_has_carla_pose(prev_fr) and _frame_has_carla_pose(next_fr)):
                        continue
                    prev_cli = _safe_int(prev_fr.get("ccli"), -1)
                    next_cli = _safe_int(next_fr.get("ccli"), -1)
                    if prev_cli < 0 or next_cli < 0 or prev_cli != next_cli:
                        continue
                    px0 = _safe_float(prev_fr.get("cx"), float("nan"))
                    py0 = _safe_float(prev_fr.get("cy"), float("nan"))
                    pyaw0 = _safe_float(prev_fr.get("cyaw"), float("nan"))
                    px1 = _safe_float(next_fr.get("cx"), float("nan"))
                    py1 = _safe_float(next_fr.get("cy"), float("nan"))
                    pyaw1 = _safe_float(next_fr.get("cyaw"), float("nan"))
                    if not (
                        math.isfinite(px0)
                        and math.isfinite(py0)
                        and math.isfinite(pyaw0)
                        and math.isfinite(px1)
                        and math.isfinite(py1)
                        and math.isfinite(pyaw1)
                    ):
                        continue
                    t0 = _safe_float(prev_fr.get("t"), float(fi - 1) * float(dt))
                    t1 = _safe_float(next_fr.get("t"), float(fi + 1) * float(dt))
                    t = _safe_float(fr.get("t"), float(fi) * float(dt))
                    if float(t1) <= float(t0) + 1e-6:
                        alpha = 0.5
                    else:
                        alpha = max(0.0, min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0))))
                    ix = (1.0 - float(alpha)) * float(px0) + float(alpha) * float(px1)
                    iy = (1.0 - float(alpha)) * float(py0) + float(alpha) * float(py1)
                    iyaw = _interp_yaw_deg(float(pyaw0), float(pyaw1), float(alpha))
                    qx = _safe_float(fr.get("sx"), _safe_float(fr.get("x"), 0.0))
                    qy = _safe_float(fr.get("sy"), _safe_float(fr.get("y"), 0.0))
                    _set_carla_pose(
                        frame=fr,
                        line_index=int(prev_cli),
                        x=float(ix),
                        y=float(iy),
                        yaw=float(iyaw),
                        dist=float(math.hypot(float(ix) - float(qx), float(iy) - float(qy))),
                        source="raw_fallback_bridge",
                        quality="none",
                    )
        print(
            "[PERF] build_dataset CARLA stage final_safety_pass: "
            f"tracks={int(final_safety_tracks)} elapsed={time.perf_counter() - t_final_safety:.2f}s"
        )

    map_min_x, map_max_x, map_min_y, map_max_y = chosen_map.bbox
    carla_map_payload: Optional[Dict[str, object]] = None
    lane_corr_summary: Dict[str, object] = {
        "enabled": False,
        "reason": "carla_projection_disabled",
    }
    if carla_context is not None:
        lane_corr_summary = dict(carla_context.get("summary", {}) or {})
        if bool(carla_context.get("enabled", False)):
            carla_bbox = carla_context.get("carla_bbox", chosen_map.bbox)
            if isinstance(carla_bbox, (list, tuple)) and len(carla_bbox) >= 4:
                map_min_x = min(float(map_min_x), float(carla_bbox[0]))
                map_max_x = max(float(map_max_x), float(carla_bbox[1]))
                map_min_y = min(float(map_min_y), float(carla_bbox[2]))
                map_max_y = max(float(map_max_y), float(carla_bbox[3]))
            carla_map_payload = {
                "name": str(carla_context.get("carla_name", "carla_map_cache")),
                "lines": list(carla_context.get("carla_lines_data", []) or []),
            }
            _ctx_align = carla_context.get("align_cfg")
            if _ctx_align:
                carla_map_payload["align_cfg"] = dict(_ctx_align)
            image_ref = str(carla_context.get("carla_image_ref", "")).strip()
            image_bounds = carla_context.get("carla_image_bounds")
            if image_ref:
                carla_map_payload["image_ref"] = image_ref
            if isinstance(image_bounds, dict):
                try:
                    carla_map_payload["image_bounds"] = {
                        "min_x": float(image_bounds.get("min_x", 0.0)),
                        "max_x": float(image_bounds.get("max_x", 0.0)),
                        "min_y": float(image_bounds.get("min_y", 0.0)),
                        "max_y": float(image_bounds.get("max_y", 0.0)),
                    }
                except Exception:
                    pass

    out = {
        "scenario_name": scenario_name,
        "map_name": chosen_map.name,
        "map_source": chosen_map.source_path,
        "map_bbox": {
            "min_x": float(map_min_x),
            "max_x": float(map_max_x),
            "min_y": float(map_min_y),
            "max_y": float(map_max_y),
        },
        "timeline": timeline,
        "lanes": lanes_data,
        "tracks": tracks,
        "lane_correspondence": lane_corr_summary,
    }
    if carla_map_payload is not None:
        out["carla_map"] = carla_map_payload
    if timing_optimization is not None:
        out["timing_optimization"] = timing_optimization
    if walker_processing is not None:
        out["walker_sidewalk_processing"] = walker_processing
    if carla_overlap_postprocess is not None:
        out["carla_overlap_postprocess"] = carla_overlap_postprocess
    total_frames_out = 0
    for tr in tracks:
        if isinstance(tr, dict):
            frames = tr.get("frames", [])
            if isinstance(frames, list):
                total_frames_out += int(len(frames))
    print(
        "[PERF] build_dataset done: "
        f"scenario={scenario_name} tracks={len(tracks)} frames={int(total_frames_out)} "
        f"elapsed={time.perf_counter() - t_build_dataset_start:.2f}s"
    )
    return out


# =============================================================================
# CLI
# =============================================================================


def _is_scenario_directory(path: Path) -> bool:
    """Check if a directory is a valid scenario directory with YAML data."""
    if not path.is_dir():
        return False
    
    # Check for numbered subdirectories (common in scenarios)
    for subdir in path.iterdir():
        if subdir.is_dir():
            # Check for numbered subdirs like -1, -2, 1, 2
            if re.fullmatch(r"-?\d+", subdir.name or ""):
                # Check if it contains YAML files
                if list_yaml_timesteps(subdir):
                    return True
    
    # Check for direct YAML files
    if list_yaml_timesteps(path):
        return True
    
    return False


def _find_scenario_directories(parent_dir: Path) -> List[Path]:
    """Find all scenario directories within a parent directory."""
    scenarios = []
    for item in sorted(parent_dir.iterdir()):
        if item.is_dir() and _is_scenario_directory(item):
            scenarios.append(item)
    return scenarios


PROCESSING_PROFILE_CHOICES: Tuple[str, ...] = (
    "current",
    "pre_overlap_baseline",
    "pre_overlap_restore",
    "pre_overlap_retro",
    "pre_overlap_recover",
    "legacy_stable",
)


def _normalize_processing_profile_name(profile: str) -> str:
    p = str(profile or "current").strip().lower()
    if p in {"default", "latest"}:
        return "current"
    return p


def _parse_compare_profiles(raw: Optional[str], default_profile: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    out: List[str] = []
    for chunk in re.split(r"[,\s]+", text):
        token = str(chunk or "").strip()
        if not token:
            continue
        normalized = _normalize_processing_profile_name(token)
        if normalized not in PROCESSING_PROFILE_CHOICES:
            allowed = ", ".join(PROCESSING_PROFILE_CHOICES)
            raise ValueError(f"Unknown profile '{token}' in --compare-profiles. Allowed: {allowed}")
        if normalized not in out:
            out.append(normalized)

    base_profile = _normalize_processing_profile_name(default_profile)
    if base_profile not in out:
        out.insert(0, base_profile)
    return out


def _restore_environment(env_snapshot: Dict[str, str]) -> None:
    os.environ.clear()
    os.environ.update(env_snapshot)


def _apply_processing_profile(profile: str, verbose: bool = False) -> Dict[str, str]:
    """
    Apply a runtime processing profile by setting environment toggles.

    Profiles:
      - current: keep all currently enabled logic.
      - pre_overlap_baseline: disable overlap-era dedup/postprocess passes so
        outputs can be compared against a pre-overlap style baseline.
      - pre_overlap_restore: reconstructed "good-before-overlap-metric" mode
        from archived metrics; keeps robust dedup enabled while disabling
        overlap reduction/parked overlap logic and pinning CARLA tuning knobs.
      - pre_overlap_retro: pre-overlap baseline + conservative pre-overlap-like
        lane transition tuning to isolate regressions from newer global knobs.
      - pre_overlap_recover: keep current lane/CARLA behavior but disable
        overlap-era CARLA postprocess/pruning passes to recover pre-overlap
        actor population and geometry fidelity.
      - legacy_stable: conservative lane-change / intersection tuning profile
        for regression debugging when current settings are overly aggressive.
    """
    p = _normalize_processing_profile_name(profile)

    applied: Dict[str, str] = {}

    def _set_env(name: str, value: object) -> None:
        sval = str(value)
        os.environ[name] = sval
        applied[name] = sval

    if p == "pre_overlap_baseline":
        _set_env("V2X_CARLA_OVERLAP_REDUCE_ENABLED", 0)
        _set_env("V2X_ENABLE_CROSS_ID_DEDUP", 0)
        _set_env("V2X_ACTOR_OVERLAP_DEDUP_ENABLED", 0)
        _set_env("V2X_EGO_ACTOR_DEDUP_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_PARKED_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_RESIDUAL_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MICRO_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_ENABLED", 0)
    elif p == "pre_overlap_restore":
        # Reconstructed checkpoint from archived reports right before overlap
        # optimization loop. Keep dedup on; disable overlap-era CARLA postprocess.
        _set_env("V2X_ENABLE_CROSS_ID_DEDUP", 1)
        _set_env("V2X_ACTOR_OVERLAP_DEDUP_ENABLED", 1)
        _set_env("V2X_EGO_ACTOR_DEDUP_ENABLED", 1)
        _set_env("V2X_CARLA_OVERLAP_REDUCE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_PARKED_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_RESIDUAL_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MICRO_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_ENABLED", 0)
        # Parked-edge anchoring came from overlap stage; disable in restore mode.
        _set_env("V2X_CARLA_PARKED_RAW_ANCHOR_ENABLED", 0)
        _set_env("V2X_CARLA_PARKED_EDGE_FREEZE_ENABLED", 0)
        # Historical checkpoint logs rebuilt correspondence when candidate rows
        # were absent in cache payloads; preserve that behavior for restore mode.
        _set_env("V2X_LANE_CORR_REFRESH_WHEN_MISSING_CANDIDATES", 1)
        # Restore CARLA tuning defaults from archived optimization baseline.
        _set_env("V2X_CARLA_OPPOSITE_REJECT_DEG", 170.0)
        _set_env("V2X_CARLA_WRONG_WAY_REJECT_DEG", 170.0)
        _set_env("V2X_CARLA_NEAREST_CONT_SCORE_SLACK", 0.60)
        _set_env("V2X_CARLA_NEAREST_CONT_DIST_SLACK", 0.80)
        _set_env("V2X_CARLA_CORR_SCORE_MARGIN_GOOD", 0.85)
        _set_env("V2X_CARLA_CORR_SCORE_MARGIN_WEAK", 0.55)
        _set_env("V2X_CARLA_SMOOTH_MAX_MID_RUN", 12)
        _set_env("V2X_CARLA_TRANSITION_SPIKE_MAX_FRAMES", 3)
        _set_env("V2X_CARLA_SMOOTH_MIN_STABLE_NEIGHBOR", 30)
        _set_env("V2X_CARLA_FAR_MAX_NEAREST", 8.0)
        _set_env("V2X_CARLA_ENABLE_MICRO_JITTER_SMOOTH", 0)
        # Conservative lane-change suppression improved regression scenarios
        # (7_0 and 14_0) without introducing wrong-way regressions.
        _set_env("V2X_ALIGN_LANE_CHANGE_JUMP_BASE_PENALTY", 360.0)
        _set_env("V2X_ALIGN_LANE_CHANGE_JUMP_PER_M_PENALTY", 220.0)
        _set_env("V2X_ALIGN_LANE_CHANGE_WEAK_EVIDENCE_PENALTY", 300.0)
        _set_env("V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M", 1.4)
        _set_env("V2X_ALIGN_LANE_CHANGE_HORIZON_MIN_GAIN_M", 1.6)
        _set_env("V2X_ALIGN_LANE_CHANGE_HORIZON_PENALTY", 260.0)
        _set_env("V2X_ALIGN_SIGN_FLIP_SMALL_STEP_PENALTY", 560.0)
        _set_env("V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY", 820.0)
    elif p == "pre_overlap_retro":
        # Start from overlap-disabled baseline.
        _set_env("V2X_CARLA_OVERLAP_REDUCE_ENABLED", 0)
        _set_env("V2X_ENABLE_CROSS_ID_DEDUP", 0)
        _set_env("V2X_ACTOR_OVERLAP_DEDUP_ENABLED", 0)
        _set_env("V2X_EGO_ACTOR_DEDUP_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_PARKED_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_RESIDUAL_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MICRO_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_ENABLED", 0)
        # Also disable the newer strict jump suppressors so this profile behaves
        # closer to the pre-overlap global alignment defaults.
        _set_env("V2X_ALIGN_ENABLE_SUPPRESS_WEAK_JUMP_LANE_CHANGES", 0)
        _set_env("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_HARD_REJECT", 0)
        _set_env("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_PENALTY", 160.0)
        _set_env("V2X_ALIGN_LANE_CHANGE_JUMP_GUARD_MIN_GAIN_M", 0.95)
        _set_env("V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY", 340.0)
        _set_env("V2X_ALIGN_ENABLE_DIRECT_TURN_BOUNDARIES", 0)
        _set_env("V2X_CARLA_HOLD_SEMANTIC_LINE_IDS", 0)
        _set_env("V2X_CARLA_TRANSITION_WINDOW_MAX_SHIFT_M", 0.85)
        _set_env("V2X_CARLA_TRANSITION_WINDOW_PASSES", 2)
    elif p == "pre_overlap_recover":
        # Keep current alignment/CARLA projection defaults, but disable overlap
        # postprocess and parked-edge anchoring that altered actor population.
        _set_env("V2X_CARLA_OVERLAP_REDUCE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_EGO_ACTOR_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_PARKED_DUP_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_RESIDUAL_PRUNE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_REMOVE_OBSTRUCTING_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MICRO_NUDGE_PARKED_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_MOVING_PAIR_NUDGE_ENABLED", 0)
        _set_env("V2X_CARLA_OVERLAP_FRAME_RAW_RESTORE_ENABLED", 0)
        _set_env("V2X_CARLA_PARKED_RAW_ANCHOR_ENABLED", 0)
        _set_env("V2X_CARLA_PARKED_EDGE_FREEZE_ENABLED", 0)
    elif p == "legacy_stable":
        # Keep all logic available, but restore conservative behavior to reduce
        # weakly-supported lane switches and intersection over-processing.
        _set_env("V2X_CARLA_OVERLAP_REDUCE_ENABLED", 0)
        _set_env("V2X_ENABLE_CROSS_ID_DEDUP", 1)
        _set_env("V2X_ACTOR_OVERLAP_DEDUP_ENABLED", 1)
        _set_env("V2X_EGO_ACTOR_DEDUP_ENABLED", 1)
        _set_env("V2X_ALIGN_LANE_CHANGE_JUMP_BASE_PENALTY", 260.0)
        _set_env("V2X_ALIGN_LANE_CHANGE_JUMP_PER_M_PENALTY", 150.0)
        _set_env("V2X_ALIGN_LANE_CHANGE_WEAK_EVIDENCE_PENALTY", 200.0)
        _set_env("V2X_ALIGN_LANE_CHANGE_MIN_GAIN_M", 1.25)
        _set_env("V2X_ALIGN_LANE_CHANGE_HORIZON_MIN_GAIN_M", 1.40)
        _set_env("V2X_ALIGN_LANE_CHANGE_HORIZON_PENALTY", 220.0)
        _set_env("V2X_ALIGN_SIGN_FLIP_SMALL_STEP_PENALTY", 460.0)
        _set_env("V2X_ALIGN_SIGN_FLIP_STRICT_PENALTY", 680.0)
        _set_env("V2X_ALIGN_ENABLE_DIRECT_TURN_BOUNDARIES", 0)
        _set_env("V2X_CARLA_HOLD_SEMANTIC_LINE_IDS", 0)
        _set_env("V2X_CARLA_TRANSITION_WINDOW_MAX_SHIFT_M", 0.85)
        _set_env("V2X_CARLA_TRANSITION_WINDOW_PASSES", 2)
    elif p == "current":
        # Preserve whatever is configured externally for the current profile.
        pass
    else:
        raise ValueError(f"Unknown processing profile: {profile}")

    if verbose:
        print(
            "[INFO] Processing profile: {} ({} env overrides)".format(
                p,
                len(applied),
            )
        )

    return applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw YAML trajectories on V2XPNP vector map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scenario_dir",
        type=str,
        help="Path to scenario directory, or parent directory containing multiple scenarios (with --multi).",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Treat scenario_dir as a parent directory containing multiple scenarios.",
    )
    parser.add_argument(
        "--processing-profile",
        choices=PROCESSING_PROFILE_CHOICES,
        default="current",
        help="Pipeline profile for before/after comparison without removing logic.",
    )
    parser.add_argument(
        "--compare-profiles",
        type=str,
        default="",
        help=(
            "Comma/space-separated processing profiles to compare for the same scenario "
            "(single-scenario mode only). Example: "
            "'current,pre_overlap_restore,legacy_stable'"
        ),
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Specific subdirectory to process, or 'all' for all subdirs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output HTML file path. Default: <scenario_dir>/trajectory_plot.html",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step between YAML frames in seconds.",
    )
    
    # Offset parameters (defaults match yaml_to_map behavior: no transform)
    parser.add_argument(
        "--tx", type=float, default=0.0,
        help="Translation offset in X direction (meters).",
    )
    parser.add_argument(
        "--ty", type=float, default=0.0,
        help="Translation offset in Y direction (meters).",
    )
    parser.add_argument(
        "--tz", type=float, default=0.0,
        help="Translation offset in Z direction (meters).",
    )
    parser.add_argument(
        "--yaw", type=float, default=0.0,
        help="Rotation offset in degrees.",
    )
    parser.add_argument(
        "--flip-y", action="store_true",
        help="Flip Y coordinates before applying rotation/translation.",
    )
    
    # Map paths
    parser.add_argument(
        "--map-pkl",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to vector map pickle files. Auto-selects best match.",
    )
    parser.add_argument(
        "--carla-projection",
        dest="carla_projection",
        action="store_true",
        help="Enable CARLA lane correspondence and trajectory projection overlay.",
    )
    parser.add_argument(
        "--no-carla-projection",
        dest="carla_projection",
        action="store_false",
        help="Disable CARLA lane correspondence / projection.",
    )
    parser.set_defaults(carla_projection=True)
    parser.add_argument(
        "--carla-map-cache",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_map_cache.pkl",
        help="Path to CARLA map cache pickle.",
    )
    parser.add_argument(
        "--carla-map-offset-json",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="Path to CARLA->V2 alignment JSON.",
    )
    parser.add_argument(
        "--carla-map-image-cache",
        type=str,
        default="/data2/marco/CoLMDriver/v2xpnp/map/carla_topdown_cache.jpg",
        help="Path to cached CARLA top-down JPEG image.",
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default="localhost",
        help="CARLA server host for optional top-down image capture.",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2005,
        help="CARLA server RPC port for optional top-down image capture.",
    )
    parser.add_argument(
        "--carla-map-name",
        type=str,
        default="ucla_v2",
        help="CARLA map name to load before optional top-down image capture.",
    )
    parser.add_argument(
        "--capture-carla-image",
        dest="capture_carla_image",
        action="store_true",
        help="Capture CARLA top-down image if cache is missing.",
    )
    parser.add_argument(
        "--no-capture-carla-image",
        dest="capture_carla_image",
        action="store_false",
        help="Disable CARLA top-down image capture; use cache only.",
    )
    parser.set_defaults(capture_carla_image=False)
    parser.add_argument(
        "--carla-topdown-method",
        type=str,
        choices=["ortho", "tiled"],
        default="ortho",
        help=(
            "How to capture the CARLA top-down image. 'ortho' (default): one wide-FOV "
            "shot orthorectified using lane waypoints as 3D ground control points "
            "(handles elevation exactly). 'tiled': legacy multi-tile near-orthographic "
            "stitch (fast, but seams + ~5%% z-elevation distortion)."
        ),
    )
    parser.add_argument(
        "--carla-topdown-altitude",
        type=float,
        default=1500.0,
        help="Camera altitude (m) for ortho capture.",
    )
    parser.add_argument(
        "--carla-topdown-fov-deg",
        type=float,
        default=60.0,
        help="Camera horizontal FOV (deg) for ortho capture.",
    )
    parser.add_argument(
        "--carla-topdown-image-px",
        type=int,
        default=16384,
        help="Captured camera image edge length (px) for ortho capture.",
    )
    parser.add_argument(
        "--carla-topdown-waypoint-spacing-m",
        type=float,
        default=1.0,
        help="Spacing (m) for the GCPs sampled from CARLA waypoints (ortho method).",
    )
    parser.add_argument(
        "--carla-topdown-px-per-meter",
        type=float,
        default=10.0,
        help="Output ortho image resolution (pixels per world meter).",
    )
    parser.add_argument(
        "--carla-topdown-gamma",
        type=float,
        default=1.25,
        help="Gamma correction applied to the captured image (>1 darkens midtones).",
    )
    parser.add_argument(
        "--lane-correspondence-top-k",
        type=int,
        default=56,
        help="Candidate CARLA lines per V2 lane during correspondence.",
    )
    parser.add_argument(
        "--lane-correspondence-cache-dir",
        type=str,
        default="v2xpnp/scripts/lane_corr_cache",
        help="Cache directory for lane correspondence.",
    )
    parser.add_argument(
        "--lane-correspondence-driving-types",
        type=str,
        default="1",
        help="Comma/space-separated V2 lane types treated as driving for one-to-one correspondence.",
    )
    parser.add_argument(
        "--carla-smooth-short-run-max-iters",
        type=int,
        default=None,
        help=(
            "Optional cap for CARLA short-run smoothing loop iterations per actor "
            "(0 or unset = unbounded)."
        ),
    )

    # Timing optimization
    parser.add_argument(
        "--maximize-safe-early-spawn",
        dest="maximize_safe_early_spawn",
        action="store_true",
        help="Advance late-detected actors to earliest safe spawn times (default: on).",
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
        help="Extra safety margin (m) for early-spawn interference checks.",
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
        help="Extra safety margin (m) for late-despawn hold checks.",
    )

    # Walker sidewalk processing
    parser.add_argument(
        "--walker-sidewalk-compression",
        dest="walker_sidewalk_compression",
        action="store_true",
        help="Enable walker sidewalk compression / classification / spawn stabilization (default: on).",
    )
    parser.add_argument(
        "--no-walker-sidewalk-compression",
        dest="walker_sidewalk_compression",
        action="store_false",
        help="Disable walker sidewalk compression pipeline.",
    )
    parser.set_defaults(walker_sidewalk_compression=True)
    parser.add_argument(
        "--walker-lane-spacing-m",
        type=float,
        default=None,
        help="Lane spacing for sidewalk geometry (default: auto-calibrate from CARLA map lines).",
    )
    parser.add_argument(
        "--walker-sidewalk-start-factor",
        type=float,
        default=0.5,
        help="Sidewalk start distance = factor * lane_spacing.",
    )
    parser.add_argument(
        "--walker-sidewalk-outer-factor",
        type=float,
        default=3.0,
        help="Sidewalk outer factor k: y = k * sidewalk_start_distance.",
    )
    parser.add_argument(
        "--walker-compression-target-band-m",
        type=float,
        default=2.5,
        help="Target sidewalk width after compression in meters.",
    )
    parser.add_argument(
        "--walker-compression-power",
        type=float,
        default=1.5,
        help="Nonlinear compression power (>1 = stronger compression at distance).",
    )
    parser.add_argument(
        "--walker-min-spawn-separation-m",
        type=float,
        default=0.8,
        help="Minimum separation between walker spawn positions.",
    )
    parser.add_argument(
        "--walker-radius-m",
        type=float,
        default=0.35,
        help="Approximate walker collision radius.",
    )
    parser.add_argument(
        "--walker-crossing-road-ratio-thresh",
        type=float,
        default=0.15,
        help="Road occupancy ratio threshold for crossing classification.",
    )
    parser.add_argument(
        "--walker-crossing-lateral-thresh-m",
        type=float,
        default=4.0,
        help="Lateral traversal distance indicating crossing behavior (m).",
    )
    parser.add_argument(
        "--walker-road-presence-min-frames",
        type=int,
        default=5,
        help="Min sustained frames in road region for crossing classification.",
    )
    parser.add_argument(
        "--walker-max-lateral-offset-m",
        type=float,
        default=3.0,
        help="Maximum allowed lateral offset from compression (m).",
    )
    parser.add_argument(
        "--export-carla-routes",
        action="store_true",
        help="Export CARLA-compatible XML route files using yaml_to_map format.",
    )
    parser.add_argument(
        "--carla-routes-dir",
        type=str,
        default=None,
        help="Output directory for CARLA routes (default: <scenario>/carla_routes, multi: <input>/carla_routes/<scenario>).",
    )
    parser.add_argument(
        "--carla-town",
        type=str,
        default="ucla_v2",
        help="CARLA town name for route XML files.",
    )
    parser.add_argument(
        "--carla-route-id",
        type=str,
        default="0",
        help="Route ID for route XML files.",
    )
    parser.add_argument(
        "--carla-ego-path-source",
        choices=("auto", "raw", "map", "corr"),
        default="auto",
        help="Ego geometry source for exported routes.",
    )
    parser.add_argument(
        "--carla-actor-control-mode",
        choices=("policy", "replay"),
        default="policy",
        help="Control mode for NPC vehicles in exported routes.",
    )
    parser.add_argument(
        "--carla-walker-control-mode",
        choices=("policy", "replay"),
        default="policy",
        help="Control mode for walkers in exported routes.",
    )
    parser.add_argument(
        "--carla-encode-timing",
        dest="carla_encode_timing",
        action="store_true",
        default=True,
        help="Include waypoint timing in exported routes (default: enabled).",
    )
    parser.add_argument(
        "--no-carla-encode-timing",
        dest="carla_encode_timing",
        action="store_false",
        help="Disable waypoint timing in exported routes.",
    )
    parser.add_argument(
        "--carla-snap-to-road",
        action="store_true",
        default=False,
        help="Enable snap_to_road for NPC routes during export.",
    )
    parser.add_argument(
        "--carla-static-spawn-only",
        action="store_true",
        default=False,
        help="For static/parked actors, export spawn-only routes.",
    )
    parser.add_argument(
        "--carla-ground-align",
        action="store_true",
        default=False,
        help="After exporting CARLA routes, connect to a running CARLA instance "
             "and use hybrid raycasting to set z/pitch/roll so vehicles sit on the road surface.",
    )
    parser.add_argument(
        "--carla-ground-align-host",
        type=str,
        default=os.environ.get("CARLA_HOST", "localhost"),
        help="CARLA server host for ground alignment (default: localhost or $CARLA_HOST).",
    )
    parser.add_argument(
        "--carla-ground-align-port",
        type=int,
        default=int(os.environ.get("CARLA_PORT", "2000")),
        help="CARLA server port for ground alignment (default: 2000 or $CARLA_PORT).",
    )
    parser.add_argument(
        "--intersection-episode-report",
        action="store_true",
        default=False,
        help="Run intersection smoothness detector and print per-episode metrics.",
    )
    parser.add_argument(
        "--intersection-episode-actor",
        type=str,
        default="all",
        help="Actor id for intersection detector, comma-separated ids, or 'all'.",
    )
    
    # Patch editor integration
    parser.add_argument(
        "--patch",
        type=str,
        default=None,
        metavar="PATCH_JSON",
        help=(
            "Path to a declarative patch JSON produced by the scenario patch editor. "
            "Applied after dataset assembly, before route export. "
            "Supports: delete, lane_segment_overrides, snap_to_outermost, "
            "waypoint_overrides, phase_override."
        ),
    )

    # Debug options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (show lane changes, alignment details).",
    )

    return parser.parse_args()


def _parse_lane_type_set(raw: str) -> List[str]:
    if raw is None:
        return ["1"]
    out: List[str] = []
    for chunk in str(raw).replace(",", " ").split():
        v = str(chunk).strip()
        if v:
            out.append(v)
    return sorted(set(out)) if out else ["1"]


def _split_tracks_for_carla_export(dataset: Dict[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ego_tracks: List[Dict[str, object]] = []
    actor_tracks: List[Dict[str, object]] = []
    tracks = dataset.get("tracks", [])
    if not isinstance(tracks, list):
        return ego_tracks, actor_tracks
    import hashlib as _hashlib
    import random as _random_mod
    from v2xpnp.pipeline.trajectory_ingest_stage_01_types_io import (
        map_obj_type as _map_obj_type,
    )

    def _stable_actor_seed(track: Dict[str, object], role: str, obj_type: str) -> int:
        seed_src = f"{track.get('id', '')}|{role}|{obj_type}"
        digest = _hashlib.sha256(seed_src.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        role = str(tr.get("role", "")).strip().lower()
        tr_copy: Dict[str, object] = copy.deepcopy(tr)
        if role == "ego":
            if not str(tr_copy.get("model", "")).strip():
                tr_copy["model"] = "vehicle.lincoln.mkz_2020"
            ego_tracks.append(tr_copy)
            continue

        if not str(tr_copy.get("model", "")).strip():
            obj_type = str(tr_copy.get("obj_type", "")).strip()
            if not obj_type:
                if role in ("walker", "pedestrian"):
                    obj_type = "pedestrian"
                elif role in ("cyclist", "bike", "bicycle"):
                    obj_type = "bicycle"
                else:
                    obj_type = "car"
            actor_rng = _random_mod.Random(_stable_actor_seed(tr_copy, role, obj_type))
            tr_copy["model"] = str(_map_obj_type(obj_type, rng=actor_rng))

        vid = _safe_int(tr_copy.get("id"), 0)
        tr_copy["vid"] = int(vid)
        tr_copy["id"] = str(tr_copy.get("id", f"actor_{vid}"))
        if role == "vehicle":
            tr_copy["parked_vehicle"] = bool(_is_parked_vehicle_track_for_overlap(tr_copy))
        else:
            tr_copy["parked_vehicle"] = False
        actor_tracks.append(tr_copy)

    return ego_tracks, actor_tracks


def _run_carla_ground_align(
    routes_dir: Path,
    host: str = "localhost",
    port: int = 2000,
    verbose: bool = False,
) -> None:
    """Run the CARLA ground-alignment postprocessor on exported routes."""
    try:
        from v2xpnp.pipeline.carla_ground_align import connect_carla, align_routes_dir
    except ImportError as exc:
        print(f"[GROUND-ALIGN] Skipping: could not import carla_ground_align: {exc}")
        return
    try:
        _client, world = connect_carla(host=host, port=port)
    except Exception as exc:
        print(f"[GROUND-ALIGN] Skipping: could not connect to CARLA at {host}:{port}: {exc}")
        return
    report = align_routes_dir(world, routes_dir, verbose=verbose)
    if "error" in report:
        print(f"[GROUND-ALIGN] Error: {report['error']}")


def _export_carla_routes_for_dataset(
    dataset: Dict[str, object],
    out_dir: Path,
    align_cfg: Dict[str, object],
    town: str,
    route_id: str,
    ego_path_source: str,
    actor_control_mode: str,
    walker_control_mode: str,
    encode_timing: bool,
    snap_to_road: bool,
    static_spawn_only: bool,
    dt: float,
) -> Dict[str, object]:
    ego_tracks, actor_tracks = _split_tracks_for_carla_export(dataset)
    if not ego_tracks and not actor_tracks:
        return {
            "enabled": False,
            "reason": "no_tracks_to_export",
            "output_dir": str(out_dir),
        }

    export_align_cfg = dict(align_cfg or {})
    lane_corr_meta = dataset.get("lane_correspondence")
    if not isinstance(lane_corr_meta, dict):
        meta = dataset.get("metadata")
        lane_corr_meta = meta.get("lane_correspondence") if isinstance(meta, dict) else None
    if isinstance(lane_corr_meta, dict):
        icp_refine = lane_corr_meta.get("icp_refine")
        if isinstance(icp_refine, dict):
            export_align_cfg["lane_corr_icp_refine"] = dict(icp_refine)

    report = ytm.export_carla_routes(
        out_dir=out_dir,
        town=str(town),
        route_id=str(route_id),
        ego_tracks=ego_tracks,
        actor_tracks=actor_tracks,
        align_cfg=export_align_cfg,
        ego_path_source=str(ego_path_source),
        actor_control_mode=str(actor_control_mode),
        walker_control_mode=str(walker_control_mode),
        encode_timing=bool(encode_timing),
        snap_to_road=bool(snap_to_road),
        static_spawn_only=bool(static_spawn_only),
        default_dt=float(dt),
    )
    dataset["carla_route_export"] = dict(report)
    return report


def _apply_timing_optimization(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    ego_trajs: Sequence[Sequence[Waypoint]],
    ego_times: Sequence[Sequence[float]],
    obj_info: Dict[int, Dict[str, object]],
    dt: float,
    maximize_safe_early_spawn: bool,
    maximize_safe_late_despawn: bool,
    early_spawn_safety_margin: float,
    late_despawn_safety_margin: float,
    verbose: bool = False,
) -> Tuple[Dict[int, List[Waypoint]], Dict[int, List[float]], Dict[str, object]]:
    timing_optimization: Dict[str, object] = {
        "timing_policy": {
            "spawn": "first_observed_frame",
            "despawn": "last_observed_frame",
            "global_early_spawn_optimization": False,
            "global_late_despawn_optimization": False,
        },
        "early_spawn": {
            "enabled": bool(maximize_safe_early_spawn),
            "applied": False,
            "reason": "disabled_by_flag" if not bool(maximize_safe_early_spawn) else "not_run",
            "adjusted_actor_ids": [],
            "adjusted_spawn_times": {},
        },
        "late_despawn": {
            "enabled": bool(maximize_safe_late_despawn),
            "applied": False,
            "reason": "disabled_by_flag" if not bool(maximize_safe_late_despawn) else "not_run",
            "adjusted_actor_ids": [],
            "hold_until_time": 0.0,
        },
    }

    if (not bool(maximize_safe_early_spawn)) and (not bool(maximize_safe_late_despawn)):
        return vehicles, vehicle_times, timing_optimization

    early_safety_margin = max(0.30, float(early_spawn_safety_margin))
    late_safety_margin = max(0.30, float(late_despawn_safety_margin))

    actor_meta_for_timing = ytm._build_actor_meta_for_timing_optimization(vehicles, obj_info)
    timing_vehicles = vehicles
    timing_vehicle_times = vehicle_times
    timing_actor_meta = actor_meta_for_timing
    timing_blocker_labels: Dict[int, str] = {}
    if actor_meta_for_timing:
        timing_vehicles, timing_vehicle_times, timing_actor_meta, timing_blocker_labels = (
            ytm._augment_timing_inputs_with_ego_blockers(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                actor_meta=actor_meta_for_timing,
                ego_trajs=ego_trajs,
                ego_times=ego_times,
                dt=float(dt),
            )
        )
        if verbose and timing_blocker_labels:
            print(
                f"[INFO] Timing optimization blockers: {len(timing_blocker_labels)} "
                "ego trajectories included as non-adjustable safety blockers."
            )

    if bool(maximize_safe_early_spawn):
        if actor_meta_for_timing:
            selected_spawn_times, early_report = ytm._maximize_safe_early_spawn_actors(
                vehicles=timing_vehicles,
                vehicle_times=timing_vehicle_times,
                actor_meta=timing_actor_meta,
                dt=float(dt),
                safety_margin=float(early_safety_margin),
            )
            selected_spawn_times = {
                int(vid): float(t)
                for vid, t in selected_spawn_times.items()
                if int(vid) in vehicles
            }
            vehicles, vehicle_times, adjusted_ids, applied_spawn_times = ytm._apply_early_spawn_time_overrides(
                vehicles=vehicles,
                vehicle_times=vehicle_times,
                early_spawn_times=selected_spawn_times,
                dt=float(dt),
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

    if bool(maximize_safe_late_despawn):
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
                selected_ids, select_report = ytm._maximize_safe_late_despawn_actors(
                    vehicles=timing_vehicles,
                    vehicle_times=timing_vehicle_times,
                    actor_meta=timing_actor_meta,
                    dt=float(dt),
                    safety_margin=float(late_safety_margin),
                    hold_until_time=float(hold_until_time),
                )
                selected_ids = {int(v) for v in selected_ids if int(v) in vehicles}
                vehicles, vehicle_times, adjusted_ids = ytm._apply_late_despawn_time_overrides(
                    vehicles=vehicles,
                    vehicle_times=vehicle_times,
                    selected_late_hold_ids=selected_ids,
                    dt=float(dt),
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

    if verbose:
        print(
            "[INFO] Timing optimization: "
            f"early_adjusted={len(timing_optimization.get('early_spawn', {}).get('adjusted_actor_ids', []))} "
            f"late_adjusted={len(timing_optimization.get('late_despawn', {}).get('adjusted_actor_ids', []))}"
        )

    return vehicles, vehicle_times, timing_optimization


def _apply_walker_sidewalk_processing(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    obj_info: Dict[int, Dict[str, object]],
    carla_runtime: Optional[Dict[str, object]],
    dt: float,
    enabled: bool,
    lane_spacing_m: Optional[float],
    sidewalk_start_factor: float,
    sidewalk_outer_factor: float,
    compression_target_band_m: float,
    compression_power: float,
    min_spawn_separation_m: float,
    walker_radius_m: float,
    crossing_road_ratio_thresh: float,
    crossing_lateral_thresh_m: float,
    road_presence_min_frames: int,
    max_lateral_offset_m: float,
    verbose: bool = False,
) -> Tuple[Dict[int, List[Waypoint]], Dict[str, object]]:
    if not bool(enabled):
        return vehicles, {"enabled": False, "reason": "disabled_by_flag"}

    lines_xy: List[List[List[float]]] = []
    if carla_runtime is not None:
        raw_lines = carla_runtime.get("lines_xy", [])
        if isinstance(raw_lines, (list, tuple)):
            for ln in raw_lines:
                if not isinstance(ln, (list, tuple)):
                    continue
                pts: List[List[float]] = []
                for p in ln:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        px = _safe_float(p[0], float("nan"))
                        py = _safe_float(p[1], float("nan"))
                        if math.isfinite(px) and math.isfinite(py):
                            pts.append([float(px), float(py)])
                if len(pts) >= 2:
                    lines_xy.append(pts)

    if not lines_xy:
        if verbose:
            print("[WARN] Walker sidewalk compression requested but no aligned CARLA map lines are available.")
        return vehicles, {"enabled": False, "reason": "no_carla_map_lines"}

    if verbose:
        print(f"[INFO] Processing walker sidewalk compression using {len(lines_xy)} CARLA road polylines...")

    def _copy_wp(wp: Waypoint) -> Waypoint:
        return Waypoint(
            x=float(wp.x),
            y=float(wp.y),
            z=float(wp.z),
            yaw=float(wp.yaw),
            pitch=float(getattr(wp, "pitch", 0.0)),
            roll=float(getattr(wp, "roll", 0.0)),
        )

    # Preserve original pedestrian trajectories so non-sidewalk walkers remain unchanged.
    original_walker_trajs: Dict[int, List[Waypoint]] = {}
    for vid, traj in vehicles.items():
        meta = obj_info.get(int(vid), {})
        obj_type = str(meta.get("obj_type") or "")
        if ytm._is_pedestrian_type(obj_type):
            original_walker_trajs[int(vid)] = [_copy_wp(wp) for wp in traj]

    walker_processor = ytm.WalkerSidewalkProcessor(
        carla_map_lines=lines_xy,
        lane_spacing_m=lane_spacing_m,
        sidewalk_start_factor=float(sidewalk_start_factor),
        sidewalk_outer_factor=float(sidewalk_outer_factor),
        compression_target_band_m=float(compression_target_band_m),
        compression_power=float(compression_power),
        min_spawn_separation_m=float(min_spawn_separation_m),
        walker_radius_m=float(walker_radius_m),
        crossing_road_ratio_thresh=float(crossing_road_ratio_thresh),
        crossing_lateral_thresh_m=float(crossing_lateral_thresh_m),
        road_presence_min_frames=int(road_presence_min_frames),
        max_lateral_offset_m=float(max_lateral_offset_m),
        dt=float(dt),
        # In visualization, keep actor cardinality stable: stationary/jitter walkers
        # remain present as static tracks instead of being dropped entirely.
        freeze_stationary_jitter=True,
    )
    updated_vehicles, report = walker_processor.process_walkers(
        vehicles=vehicles,
        vehicle_times=vehicle_times,
        obj_info=obj_info,
    )

    # Enforce plot-view behavior: only sidewalk-consistent walkers may be altered.
    # Road/crossing/jaywalking/unclassified walkers are restored to raw trajectories.
    restored_non_sidewalk: List[int] = []
    classifications = report.get("classifications", {})
    if isinstance(classifications, dict):
        for raw_vid, cls in classifications.items():
            vid = _safe_int(raw_vid, -1)
            if vid < 0 or vid not in original_walker_trajs:
                continue
            if not isinstance(cls, dict):
                continue
            is_sidewalk = bool(cls.get("is_sidewalk_consistent", False))
            if is_sidewalk:
                continue
            updated_vehicles[int(vid)] = [_copy_wp(wp) for wp in original_walker_trajs[int(vid)]]
            restored_non_sidewalk.append(int(vid))

    # Keep report consistent with restored trajectories.
    if restored_non_sidewalk:
        restored_set = {int(v) for v in restored_non_sidewalk}
        compressions = report.get("compressions")
        if isinstance(compressions, dict):
            for vid in restored_non_sidewalk:
                row = compressions.get(int(vid))
                if row is None:
                    row = compressions.get(str(int(vid)))
                if isinstance(row, dict):
                    row["applied"] = False
                    row["avg_lateral_offset"] = 0.0
                    row["max_lateral_offset"] = 0.0
                    row["frames_modified"] = 0
                    row["reason"] = "restored_non_sidewalk_walker_raw"

            applied_rows = [
                v for v in compressions.values()
                if isinstance(v, dict) and bool(v.get("applied", False))
            ]
            report["compression_summary"] = {
                "compressed": int(len(applied_rows)),
                "skipped": int(max(0, len(compressions) - len(applied_rows))),
                "total_frames_modified": int(
                    sum(int(_safe_int(v.get("frames_modified", 0), 0)) for v in compressions.values() if isinstance(v, dict))
                ),
                "avg_lateral_offset": float(
                    np.mean([float(_safe_float(v.get("avg_lateral_offset", 0.0), 0.0)) for v in applied_rows])
                ) if applied_rows else 0.0,
                "max_lateral_offset": float(
                    np.max([float(_safe_float(v.get("max_lateral_offset", 0.0), 0.0)) for v in applied_rows])
                ) if applied_rows else 0.0,
            }

        stabilization = report.get("stabilization")
        if isinstance(stabilization, dict):
            details = stabilization.get("details")
            if isinstance(details, dict):
                for vid in restored_non_sidewalk:
                    details.pop(int(vid), None)
                    details.pop(str(int(vid)), None)
                offset_mags: List[float] = []
                for row in details.values():
                    if isinstance(row, dict):
                        offset_mags.append(float(_safe_float(row.get("offset_magnitude", 0.0), 0.0)))
                stabilization["adjusted_count"] = int(len(offset_mags))
                stabilization["avg_separation_offset"] = float(np.mean(offset_mags)) if offset_mags else 0.0
                stabilization["max_separation_offset"] = float(np.max(offset_mags)) if offset_mags else 0.0

        report["restored_non_sidewalk_walkers"] = sorted(int(v) for v in restored_set)

    if verbose:
        if report.get("walker_count", 0) > 0:
            stationary_removed = report.get("stationary_removed_count", 0)
            stationary_frozen = report.get("stationary_frozen_count", 0)
            stationary_preserved = report.get("stationary_preserved_count", 0)
            cls_summary = report.get("classification_summary", {})
            comp_summary = report.get("compression_summary", {})
            stab = report.get("stabilization", {})
            restored_cnt = len(report.get("restored_non_sidewalk_walkers", []) or [])
            print(
                f"[INFO] Walker processing: {report.get('walker_count', 0)} walkers | "
                f"stationary/jitter removed: {stationary_removed}, frozen_static: {stationary_frozen}, "
                f"preserved_smoothed: {stationary_preserved} | "
                f"classified: sidewalk={cls_summary.get('sidewalk_consistent', 0)}, "
                f"crossing={cls_summary.get('crossing', 0)}, "
                f"jaywalking={cls_summary.get('jaywalking', 0)}, "
                f"road_walking={cls_summary.get('road_walking', 0)} | "
                f"compressed: {comp_summary.get('compressed', 0)}, "
                f"spawn-stabilized: {stab.get('adjusted_count', 0)} | "
                f"restored_non_sidewalk={restored_cnt}"
            )
        else:
            print("[INFO] Walker processing: no walkers found in trajectory data.")

    return updated_vehicles, report


def process_single_scenario(
    scenario_dir: Path,
    map_data_list: List[VectorMapData],
    dt: float,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    flip_y: bool,
    subdir: Optional[str] = None,
    carla_runtime: Optional[Dict[str, object]] = None,
    enable_carla_projection: bool = True,
    timing_cfg: Optional[Dict[str, object]] = None,
    walker_cfg: Optional[Dict[str, object]] = None,
    carla_context_cache: Optional[Dict[str, Dict[str, object]]] = None,
    intersection_episode_report: bool = False,
    intersection_episode_report_actor: str = "all",
    verbose: bool = True,
) -> Optional[Dict[str, object]]:
    """Process a single scenario and return its dataset."""
    try:
        yaml_dirs = pick_yaml_dirs(scenario_dir, subdir)
    except SystemExit:
        if verbose:
            print(f"[WARN] No YAML directories found in: {scenario_dir.name}")
        return None
    
    if not yaml_dirs:
        if verbose:
            print(f"[WARN] No YAML directories in: {scenario_dir.name}")
        return None

    if verbose:
        print(f"[INFO] Processing: {scenario_dir.name} ({len(yaml_dirs)} subdirs)")

    # Load trajectories
    vehicles, vehicle_times, ego_trajs, ego_times, obj_info = load_trajectories(
        yaml_dirs=yaml_dirs,
        dt=dt,
        tx=tx,
        ty=ty,
        tz=tz,
        yaw_deg=yaw_deg,
        flip_y=flip_y,
    )

    if not ego_trajs and not vehicles:
        if verbose:
            print(f"[WARN] No trajectories found in: {scenario_dir.name}")
        return None

    # Timing optimization (safe early spawn / late despawn).
    timing_optimization: Dict[str, object] = {
        "timing_policy": {
            "spawn": "first_observed_frame",
            "despawn": "last_observed_frame",
            "global_early_spawn_optimization": False,
            "global_late_despawn_optimization": False,
        },
        "early_spawn": {"enabled": False, "applied": False, "reason": "disabled"},
        "late_despawn": {"enabled": False, "applied": False, "reason": "disabled"},
    }
    if timing_cfg is not None:
        vehicles, vehicle_times, timing_optimization = _apply_timing_optimization(
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            ego_trajs=ego_trajs,
            ego_times=ego_times,
            obj_info=obj_info,
            dt=float(dt),
            maximize_safe_early_spawn=bool(timing_cfg.get("maximize_safe_early_spawn", True)),
            maximize_safe_late_despawn=bool(timing_cfg.get("maximize_safe_late_despawn", True)),
            early_spawn_safety_margin=float(timing_cfg.get("early_spawn_safety_margin", 0.25)),
            late_despawn_safety_margin=float(timing_cfg.get("late_despawn_safety_margin", 0.25)),
            verbose=bool(verbose),
        )
        # Post-timing handoff dedup: early-spawn extension can create handoff pairs
        # that were invisible at overlap-dedup time.  Remove predecessor actors now.
        vehicles, vehicle_times, obj_info = _apply_post_timing_handoff_dedup(
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            obj_info=obj_info,
        )

    # Select best map
    try:
        chosen_map, _ = select_best_map(
            maps=map_data_list,
            ego_trajs=ego_trajs,
            vehicles=vehicles,
        )
    except RuntimeError as e:
        if verbose:
            print(f"[WARN] Map selection failed for {scenario_dir.name}: {e}")
        return None

    # Walker sidewalk compression / classification / spawn stabilization.
    walker_processing_report: Dict[str, object] = {"enabled": False, "reason": "disabled_by_flag"}
    if walker_cfg is not None:
        vehicles, walker_processing_report = _apply_walker_sidewalk_processing(
            vehicles=vehicles,
            vehicle_times=vehicle_times,
            obj_info=obj_info,
            carla_runtime=carla_runtime,
            dt=float(dt),
            enabled=bool(walker_cfg.get("enabled", False)),
            lane_spacing_m=(
                float(walker_cfg["lane_spacing_m"])
                if walker_cfg.get("lane_spacing_m", None) is not None
                else None
            ),
            sidewalk_start_factor=float(walker_cfg.get("sidewalk_start_factor", 0.5)),
            sidewalk_outer_factor=float(walker_cfg.get("sidewalk_outer_factor", 3.0)),
            compression_target_band_m=float(walker_cfg.get("compression_target_band_m", 2.5)),
            compression_power=float(walker_cfg.get("compression_power", 1.5)),
            min_spawn_separation_m=float(walker_cfg.get("min_spawn_separation_m", 0.8)),
            walker_radius_m=float(walker_cfg.get("walker_radius_m", 0.35)),
            crossing_road_ratio_thresh=float(walker_cfg.get("crossing_road_ratio_thresh", 0.15)),
            crossing_lateral_thresh_m=float(walker_cfg.get("crossing_lateral_thresh_m", 4.0)),
            road_presence_min_frames=int(walker_cfg.get("road_presence_min_frames", 5)),
            max_lateral_offset_m=float(walker_cfg.get("max_lateral_offset_m", 3.0)),
            verbose=bool(verbose),
        )

    # Create LaneMatcher for snapping
    matcher = LaneMatcher(chosen_map)
    if verbose:
        n_vehicles = len(vehicles)
        n_frames = sum(len(t) for t in vehicles.values())
        print(f"[INFO] Starting lane alignment + CARLA projection: {n_vehicles} actors, {n_frames} total frames...")

    # Build or fetch CARLA projection context for selected map.
    carla_context: Optional[Dict[str, object]] = None
    if bool(enable_carla_projection) and carla_runtime is not None:
        cache_key = str(chosen_map.name)
        if carla_context_cache is not None and cache_key in carla_context_cache:
            carla_context = carla_context_cache[cache_key]
        else:
            try:
                carla_context = build_carla_projection_context(
                    chosen_map=chosen_map,
                    carla_lines_xy=carla_runtime.get("lines_xy", []),
                    carla_line_records=carla_runtime.get("line_records", []),
                    carla_bbox=carla_runtime.get("bbox", chosen_map.bbox),
                    carla_source_path=str(carla_runtime.get("source_path", "")),
                    carla_name=str(carla_runtime.get("map_name", "carla_map_cache")),
                    lane_corr_top_k=int(carla_runtime.get("lane_corr_top_k", 56)),
                    lane_corr_cache_dir=carla_runtime.get("lane_corr_cache_dir"),
                    lane_corr_driving_types=carla_runtime.get("lane_corr_driving_types", ["1"]),
                    verbose=bool(verbose),
                )
            except Exception as exc:
                carla_context = {
                    "enabled": False,
                    "summary": {
                        "enabled": False,
                        "reason": f"lane_correspondence_failed: {exc}",
                        "map_name": str(chosen_map.name),
                    },
                }
                if verbose:
                    print(f"[WARN] CARLA correspondence failed for {chosen_map.name}: {exc}")
            if carla_context_cache is not None:
                carla_context_cache[cache_key] = dict(carla_context or {})

    if carla_context is not None and carla_runtime is not None:
        image_ref = str(carla_runtime.get("image_ref", "")).strip()
        image_bounds = carla_runtime.get("image_bounds")
        if image_ref:
            carla_context["carla_image_ref"] = image_ref
        if isinstance(image_bounds, dict):
            try:
                carla_context["carla_image_bounds"] = {
                    "min_x": float(image_bounds.get("min_x", 0.0)),
                    "max_x": float(image_bounds.get("max_x", 0.0)),
                    "min_y": float(image_bounds.get("min_y", 0.0)),
                    "max_y": float(image_bounds.get("max_y", 0.0)),
                }
            except Exception:
                pass
        _rt_align = carla_runtime.get("align_cfg")
        if _rt_align:
            carla_context["align_cfg"] = dict(_rt_align)

    # Build dataset
    print(f"[PERF] scenario={scenario_dir.name} build_dataset start")
    t_dataset_build = time.perf_counter()
    dataset = build_dataset(
        scenario_name=scenario_dir.name,
        chosen_map=chosen_map,
        ego_trajs=ego_trajs,
        ego_times=ego_times,
        vehicles=vehicles,
        vehicle_times=vehicle_times,
        obj_info=obj_info,
        dt=dt,
        matcher=matcher,
        carla_context=carla_context,
        timing_optimization=timing_optimization,
        walker_processing=walker_processing_report,
        verbose=verbose,
    )
    print(
        "[PERF] scenario={} build_dataset done: elapsed={:.2f}s".format(
            scenario_dir.name,
            time.perf_counter() - t_dataset_build,
        )
    )

    if bool(intersection_episode_report):
        t_intersection_report = time.perf_counter()
        report_intersection_episode_quality(
            tracks=list(dataset.get("tracks", []) or []),
            carla_context=carla_context,
            actor_filter=str(intersection_episode_report_actor),
            scenario_name=str(scenario_dir.name),
        )
        print(
            "[PERF] scenario={} intersection_episode_report done: elapsed={:.2f}s".format(
                scenario_dir.name,
                time.perf_counter() - t_intersection_report,
            )
        )
    
    if verbose:
        print(f"  - {len(ego_trajs)} egos, {len(vehicles)} actors, map={chosen_map.name}")

    return dataset


def _apply_patch_file(
    dataset: Dict[str, object],
    patch_path: Path,
    verbose: bool = False,
) -> None:
    """Load and apply a patch JSON file to dataset in-place (no-op if file missing)."""
    if not patch_path.exists():
        return
    try:
        patch_data = json.loads(patch_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[PATCH] ERROR reading {patch_path}: {exc}")
        return
    try:
        import sys as _sys
        import os as _os
        # Add repo root to path so tools.patch_editor is importable regardless of CWD
        _repo_root = str(Path(__file__).resolve().parents[2])
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from tools.patch_editor.patch_apply import apply_patch_to_dataset
        n = apply_patch_to_dataset(dataset, patch_data, verbose=verbose)
        if n > 0 or verbose:
            print(f"[PATCH] Applied {n} override(s) from {patch_path.name}")
    except Exception as exc:
        print(f"[PATCH] ERROR applying patch {patch_path}: {exc}")


def main() -> None:
    args = parse_args()
    if args.carla_smooth_short_run_max_iters is not None:
        cap = int(args.carla_smooth_short_run_max_iters)
        if cap < 0:
            raise SystemExit("--carla-smooth-short-run-max-iters must be >= 0")
        if cap == 0:
            os.environ.pop("V2X_CARLA_SMOOTH_SHORT_RUN_MAX_ITERS", None)
            print("[INFO] CARLA short-run smoothing iteration cap: unbounded")
        else:
            os.environ["V2X_CARLA_SMOOTH_SHORT_RUN_MAX_ITERS"] = str(int(cap))
            print(f"[INFO] CARLA short-run smoothing iteration cap: {int(cap)}")
    base_env_snapshot = dict(os.environ)
    try:
        compare_profiles = _parse_compare_profiles(
            raw=args.compare_profiles,
            default_profile=str(args.processing_profile),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    if bool(args.multi) and compare_profiles:
        raise SystemExit("--compare-profiles cannot be used with --multi.")

    profile_overrides: Dict[str, str] = {}
    if compare_profiles:
        print(f"[INFO] Profile compare mode: {compare_profiles}")
    else:
        profile_overrides = _apply_processing_profile(
            profile=str(args.processing_profile),
            verbose=bool(args.verbose),
        )

    input_dir = Path(args.scenario_dir).expanduser().resolve()
    if not input_dir.exists():
        raise SystemExit(f"Directory not found: {input_dir}")

    # Load maps first (shared across all scenarios)
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

    print("[INFO] Loading maps...")
    map_data_list = [load_vector_map(p) for p in map_paths]
    print(f"[INFO] Loaded {len(map_data_list)} maps")

    carla_runtime: Optional[Dict[str, object]] = None
    carla_image_store: Dict[str, Dict[str, object]] = {}
    carla_context_cache: Dict[str, Dict[str, object]] = {}
    need_carla_lines = bool(args.carla_projection) or bool(args.walker_sidewalk_compression)
    if need_carla_lines:
        carla_cache_path = Path(args.carla_map_cache).expanduser().resolve()
        align_path = (
            Path(args.carla_map_offset_json).expanduser().resolve()
            if str(args.carla_map_offset_json).strip()
            else None
        )
        if not carla_cache_path.exists():
            print(f"[WARN] CARLA map cache not found; disabling CARLA projection: {carla_cache_path}")
        else:
            try:
                raw_lines, raw_bounds, carla_map_name, raw_line_records = ytm._load_carla_map_cache(carla_cache_path)
                align_cfg = ytm._load_carla_alignment_cfg(align_path)
                transformed_lines, transformed_bbox = ytm._transform_carla_lines(raw_lines, align_cfg)
                transformed_line_records = ytm._transform_carla_line_records(raw_line_records, align_cfg)
                if not transformed_lines:
                    print("[WARN] No aligned CARLA lines available; disabling CARLA projection.")
                else:
                    lane_corr_cache_dir: Optional[Path] = None
                    if str(args.lane_correspondence_cache_dir).strip():
                        lane_corr_cache_dir = Path(args.lane_correspondence_cache_dir).expanduser().resolve()
                    lane_corr_driving_types = _parse_lane_type_set(args.lane_correspondence_driving_types)
                    carla_runtime = {
                        "lines_xy": transformed_lines,
                        "line_records": transformed_line_records,
                        "bbox": transformed_bbox,
                        "source_path": str(carla_cache_path),
                        "map_name": str(carla_map_name or "carla_map_cache"),
                        "lane_corr_top_k": int(args.lane_correspondence_top_k),
                        "lane_corr_cache_dir": lane_corr_cache_dir,
                        "lane_corr_driving_types": lane_corr_driving_types,
                        "raw_bounds": raw_bounds,
                        "align_cfg": dict(align_cfg),
                    }
                    enabled_features: List[str] = []
                    if bool(args.carla_projection):
                        enabled_features.append("carla_projection")
                    if bool(args.walker_sidewalk_compression):
                        enabled_features.append("walker_sidewalk_processing")
                    print(
                        "[INFO] Loaded aligned CARLA map geometry: "
                        f"lines={len(transformed_lines)} cache={carla_cache_path} "
                        f"align={align_path if align_path else '-'} "
                        f"features={enabled_features} driving_types={lane_corr_driving_types}"
                    )

                    image_cache_path = Path(args.carla_map_image_cache).expanduser().resolve()
                    image_meta_path = image_cache_path.with_suffix(".json")
                    raw_bounds_tuple: Optional[Tuple[float, float, float, float]] = None
                    if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) >= 4:
                        try:
                            raw_bounds_tuple = (
                                float(raw_bounds[0]),
                                float(raw_bounds[1]),
                                float(raw_bounds[2]),
                                float(raw_bounds[3]),
                            )
                        except Exception:
                            raw_bounds_tuple = None

                    topdown_result = ytm._load_or_capture_carla_topdown(
                        image_cache_path=image_cache_path,
                        meta_cache_path=image_meta_path,
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
                        ortho_gamma=float(args.carla_topdown_gamma),
                    )
                    if topdown_result is not None:
                        jpeg_bytes, img_raw_bounds = topdown_result
                        image_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                        image_bounds = ytm._transform_image_bounds_to_v2xpnp(
                            raw_bounds=img_raw_bounds,
                            align_cfg=align_cfg,
                        )
                        image_ref = str(carla_map_name or "carla_map_cache")
                        carla_runtime["image_ref"] = image_ref
                        carla_runtime["image_bounds"] = dict(image_bounds)
                        carla_image_store[image_ref] = {
                            "image_b64": image_b64,
                            "image_bounds": dict(image_bounds),
                        }
                        print(
                            "[INFO] CARLA top-down image attached: "
                            f"ref={image_ref} bytes={len(jpeg_bytes)} "
                            f"b64={len(image_b64)} bounds={image_bounds}"
                        )
                    else:
                        print("[INFO] No CARLA top-down image available (cache missing or capture disabled/failed).")
            except Exception as exc:
                print(f"[WARN] Failed to initialize CARLA map geometry: {exc}")

    timing_cfg = {
        "maximize_safe_early_spawn": bool(args.maximize_safe_early_spawn),
        "maximize_safe_late_despawn": bool(args.maximize_safe_late_despawn),
        "early_spawn_safety_margin": float(args.early_spawn_safety_margin),
        "late_despawn_safety_margin": float(args.late_despawn_safety_margin),
    }
    walker_cfg = {
        "enabled": bool(args.walker_sidewalk_compression),
        "lane_spacing_m": args.walker_lane_spacing_m,
        "sidewalk_start_factor": float(args.walker_sidewalk_start_factor),
        "sidewalk_outer_factor": float(args.walker_sidewalk_outer_factor),
        "compression_target_band_m": float(args.walker_compression_target_band_m),
        "compression_power": float(args.walker_compression_power),
        "min_spawn_separation_m": float(args.walker_min_spawn_separation_m),
        "walker_radius_m": float(args.walker_radius_m),
        "crossing_road_ratio_thresh": float(args.walker_crossing_road_ratio_thresh),
        "crossing_lateral_thresh_m": float(args.walker_crossing_lateral_thresh_m),
        "road_presence_min_frames": int(args.walker_road_presence_min_frames),
        "max_lateral_offset_m": float(args.walker_max_lateral_offset_m),
    }

    if args.multi:
        # Multi-scenario mode
        scenario_dirs = _find_scenario_directories(input_dir)
        if not scenario_dirs:
            raise SystemExit(f"No scenario directories found in: {input_dir}")

        print(f"[INFO] Found {len(scenario_dirs)} scenarios in: {input_dir.name}")
        
        # Process all scenarios
        all_datasets: List[Dict[str, object]] = []
        for scenario_dir in scenario_dirs:
            dataset = process_single_scenario(
                scenario_dir=scenario_dir,
                map_data_list=map_data_list,
                dt=float(args.dt),
                tx=float(args.tx),
                ty=float(args.ty),
                tz=float(args.tz),
                yaw_deg=float(args.yaw),
                flip_y=bool(args.flip_y),
                subdir=args.subdir,
                carla_runtime=carla_runtime,
                enable_carla_projection=bool(args.carla_projection),
                timing_cfg=timing_cfg,
                walker_cfg=walker_cfg,
                carla_context_cache=carla_context_cache,
                intersection_episode_report=bool(args.intersection_episode_report),
                intersection_episode_report_actor=str(args.intersection_episode_actor),
                verbose=args.verbose,
            )
            if dataset:
                # Apply per-scenario patch if present alongside the HTML output
                _apply_patch_file(
                    dataset,
                    scenario_dir / "trajectory_plot.patch.json",
                    verbose=bool(args.verbose),
                )
                dataset["processing_profile"] = {
                    "name": str(args.processing_profile),
                    "env_overrides": dict(profile_overrides),
                }
                if bool(args.export_carla_routes):
                    if carla_runtime is None:
                        dataset["carla_route_export"] = {
                            "enabled": False,
                            "reason": "missing_carla_alignment",
                        }
                        print(
                            f"[WARN] Skipping CARLA route export for {scenario_dir.name}: "
                            "missing CARLA runtime/alignment."
                        )
                    else:
                        align_cfg = dict(carla_runtime.get("align_cfg", {}) or {})
                        if not align_cfg:
                            dataset["carla_route_export"] = {
                                "enabled": False,
                                "reason": "missing_carla_alignment",
                            }
                            print(
                                f"[WARN] Skipping CARLA route export for {scenario_dir.name}: "
                                "alignment config missing."
                            )
                        else:
                            if args.carla_routes_dir:
                                routes_root = Path(args.carla_routes_dir).expanduser().resolve() / scenario_dir.name
                            else:
                                routes_root = input_dir / "carla_routes" / scenario_dir.name
                            export_report = _export_carla_routes_for_dataset(
                                dataset=dataset,
                                out_dir=routes_root,
                                align_cfg=align_cfg,
                                town=str(args.carla_town),
                                route_id=str(args.carla_route_id),
                                ego_path_source=str(args.carla_ego_path_source),
                                actor_control_mode=str(args.carla_actor_control_mode),
                                walker_control_mode=str(args.carla_walker_control_mode),
                                encode_timing=bool(args.carla_encode_timing),
                                snap_to_road=bool(args.carla_snap_to_road),
                                static_spawn_only=bool(args.carla_static_spawn_only),
                                dt=float(args.dt),
                            )
                            if bool(export_report.get("enabled", False)):
                                print(
                                    f"[INFO] Exported CARLA routes for {scenario_dir.name}: "
                                    f"{routes_root}"
                                )
                                if bool(args.carla_ground_align):
                                    _run_carla_ground_align(
                                        routes_root,
                                        host=str(args.carla_ground_align_host),
                                        port=int(args.carla_ground_align_port),
                                        verbose=bool(args.verbose),
                                    )
                all_datasets.append(dataset)

        if not all_datasets:
            raise SystemExit("No valid scenarios found.")

        print(f"[INFO] Successfully processed {len(all_datasets)} scenarios")

        # Build combined dataset for multi-mode
        combined_dataset = {
            "scenarios": all_datasets,
            "processing_profile": {
                "name": str(args.processing_profile),
                "env_overrides": dict(profile_overrides),
            },
        }
        if carla_image_store:
            combined_dataset["carla_images"] = carla_image_store

        # Determine output path
        if args.out:
            out_path = Path(args.out).expanduser().resolve()
        else:
            if str(args.processing_profile) == "current":
                default_name = "trajectories_multi.html"
            else:
                default_name = f"trajectories_multi_{args.processing_profile}.html"
            out_path = input_dir / default_name

        # Generate HTML
        html_content = _build_html(combined_dataset, multi_mode=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content, encoding="utf-8")
        print(f"[OK] Wrote: {out_path} ({len(all_datasets)} scenarios)")

    else:
        # Single scenario mode
        if compare_profiles:
            variant_datasets: List[Dict[str, object]] = []
            for profile_name in compare_profiles:
                print(f"[INFO] Processing profile variant: {profile_name}")
                _restore_environment(base_env_snapshot)
                variant_overrides = _apply_processing_profile(
                    profile=str(profile_name),
                    verbose=bool(args.verbose),
                )
                variant_cache: Dict[str, Dict[str, object]] = {}
                dataset_variant = process_single_scenario(
                    scenario_dir=input_dir,
                    map_data_list=map_data_list,
                    dt=float(args.dt),
                    tx=float(args.tx),
                    ty=float(args.ty),
                    tz=float(args.tz),
                    yaw_deg=float(args.yaw),
                    flip_y=bool(args.flip_y),
                    subdir=args.subdir,
                    carla_runtime=carla_runtime,
                    enable_carla_projection=bool(args.carla_projection),
                    timing_cfg=timing_cfg,
                    walker_cfg=walker_cfg,
                    carla_context_cache=variant_cache,
                    intersection_episode_report=bool(args.intersection_episode_report),
                    intersection_episode_report_actor=str(args.intersection_episode_actor),
                    verbose=args.verbose,
                )
                if not dataset_variant:
                    print(f"[WARN] Skipping profile {profile_name}: scenario processing failed.")
                    continue

                base_name = str(dataset_variant.get("scenario_name", input_dir.name))
                dataset_variant["base_scenario_name"] = base_name
                dataset_variant["scenario_name"] = f"{base_name} [{profile_name}]"
                dataset_variant["processing_profile"] = {
                    "name": str(profile_name),
                    "env_overrides": dict(variant_overrides),
                }

                if bool(args.export_carla_routes):
                    if carla_runtime is None:
                        dataset_variant["carla_route_export"] = {
                            "enabled": False,
                            "reason": "missing_carla_alignment",
                        }
                        print(
                            f"[WARN] Skipping CARLA route export for profile {profile_name}: "
                            "missing CARLA runtime/alignment."
                        )
                    else:
                        align_cfg = dict(carla_runtime.get("align_cfg", {}) or {})
                        if not align_cfg:
                            dataset_variant["carla_route_export"] = {
                                "enabled": False,
                                "reason": "missing_carla_alignment",
                            }
                            print(
                                f"[WARN] Skipping CARLA route export for profile {profile_name}: "
                                "alignment config missing."
                            )
                        else:
                            if args.carla_routes_dir:
                                routes_root = Path(args.carla_routes_dir).expanduser().resolve() / str(profile_name)
                            else:
                                routes_root = input_dir / "carla_routes" / str(profile_name)
                            export_report = _export_carla_routes_for_dataset(
                                dataset=dataset_variant,
                                out_dir=routes_root,
                                align_cfg=align_cfg,
                                town=str(args.carla_town),
                                route_id=str(args.carla_route_id),
                                ego_path_source=str(args.carla_ego_path_source),
                                actor_control_mode=str(args.carla_actor_control_mode),
                                walker_control_mode=str(args.carla_walker_control_mode),
                                encode_timing=bool(args.carla_encode_timing),
                                snap_to_road=bool(args.carla_snap_to_road),
                                static_spawn_only=bool(args.carla_static_spawn_only),
                                dt=float(args.dt),
                            )
                            if bool(export_report.get("enabled", False)):
                                print(
                                    f"[INFO] Exported CARLA routes for profile {profile_name}: "
                                    f"{routes_root}"
                                )
                                if bool(args.carla_ground_align):
                                    _run_carla_ground_align(
                                        routes_root,
                                        host=str(args.carla_ground_align_host),
                                        port=int(args.carla_ground_align_port),
                                        verbose=bool(args.verbose),
                                    )

                variant_datasets.append(dataset_variant)

            _restore_environment(base_env_snapshot)
            if not variant_datasets:
                raise SystemExit("Failed to process scenario for all compare profiles.")

            applied_profiles = [
                str(row.get("processing_profile", {}).get("name", ""))
                for row in variant_datasets
            ]
            combined_dataset: Dict[str, object] = {
                "scenarios": variant_datasets,
                "comparison_mode": {
                    "kind": "profiles",
                    "base_scenario": str(input_dir.name),
                    "profiles": [p for p in applied_profiles if p],
                },
            }
            if carla_image_store:
                combined_dataset["carla_images"] = carla_image_store

            if args.out:
                out_path = Path(args.out).expanduser().resolve()
            else:
                default_name = "trajectory_plot_compare_profiles.html"
                out_path = input_dir / default_name

            html_content = _build_html(combined_dataset, multi_mode=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html_content, encoding="utf-8")
            print(f"[OK] Wrote: {out_path} ({len(variant_datasets)} profile variants)")
            return

        dataset = process_single_scenario(
            scenario_dir=input_dir,
            map_data_list=map_data_list,
            dt=float(args.dt),
            tx=float(args.tx),
            ty=float(args.ty),
            tz=float(args.tz),
            yaw_deg=float(args.yaw),
            flip_y=bool(args.flip_y),
            subdir=args.subdir,
            carla_runtime=carla_runtime,
            enable_carla_projection=bool(args.carla_projection),
            timing_cfg=timing_cfg,
            walker_cfg=walker_cfg,
            carla_context_cache=carla_context_cache,
            intersection_episode_report=bool(args.intersection_episode_report),
            intersection_episode_report_actor=str(args.intersection_episode_actor),
            verbose=args.verbose,
        )

        if not dataset:
            raise SystemExit("Failed to process scenario.")

        # Apply patch: explicit --patch arg takes priority; fall back to
        # auto-discovered trajectory_plot.patch.json in the scenario dir.
        _patch_path = (
            Path(args.patch).expanduser().resolve()
            if args.patch
            else input_dir / "trajectory_plot.patch.json"
        )
        _apply_patch_file(dataset, _patch_path, verbose=bool(args.verbose))

        dataset["processing_profile"] = {
            "name": str(args.processing_profile),
            "env_overrides": dict(profile_overrides),
        }

        if bool(args.export_carla_routes):
            if carla_runtime is None:
                dataset["carla_route_export"] = {
                    "enabled": False,
                    "reason": "missing_carla_alignment",
                }
                print("[WARN] Skipping CARLA route export: missing CARLA runtime/alignment.")
            else:
                align_cfg = dict(carla_runtime.get("align_cfg", {}) or {})
                if not align_cfg:
                    dataset["carla_route_export"] = {
                        "enabled": False,
                        "reason": "missing_carla_alignment",
                    }
                    print("[WARN] Skipping CARLA route export: alignment config missing.")
                else:
                    routes_root = (
                        Path(args.carla_routes_dir).expanduser().resolve()
                        if args.carla_routes_dir
                        else input_dir / "carla_routes"
                    )
                    export_report = _export_carla_routes_for_dataset(
                        dataset=dataset,
                        out_dir=routes_root,
                        align_cfg=align_cfg,
                        town=str(args.carla_town),
                        route_id=str(args.carla_route_id),
                        ego_path_source=str(args.carla_ego_path_source),
                        actor_control_mode=str(args.carla_actor_control_mode),
                        walker_control_mode=str(args.carla_walker_control_mode),
                        encode_timing=bool(args.carla_encode_timing),
                        snap_to_road=bool(args.carla_snap_to_road),
                        static_spawn_only=bool(args.carla_static_spawn_only),
                        dt=float(args.dt),
                    )
                    if bool(export_report.get("enabled", False)):
                        print(f"[INFO] Exported CARLA routes: {routes_root}")
                        if bool(args.carla_ground_align):
                            _run_carla_ground_align(
                                routes_root,
                                host=str(args.carla_ground_align_host),
                                port=int(args.carla_ground_align_port),
                                verbose=bool(args.verbose),
                            )

        if carla_image_store:
            dataset["carla_images"] = carla_image_store

        # Determine output path
        if args.out:
            out_path = Path(args.out).expanduser().resolve()
        else:
            if str(args.processing_profile) == "current":
                default_name = "trajectory_plot.html"
            else:
                default_name = f"trajectory_plot_{args.processing_profile}.html"
            out_path = input_dir / default_name

        # Generate HTML
        html_content = _build_html(dataset, multi_mode=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content, encoding="utf-8")
        print(f"[OK] Wrote: {out_path}")
