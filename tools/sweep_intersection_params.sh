#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCENARIO_ROOT="/data2/marco/CoLMDriver/v2xpnp/dataset/train4"
OUT_DIR="debug_runs/intersection_sweep_$(date -u +%Y%m%d_%H%M%S)"
MAX_SCENARIOS=0
SCENARIO_NAMES=()

usage() {
  cat <<'USAGE'
Usage:
  tools/sweep_intersection_params.sh [options]

Options:
  --scenario-root PATH      Scenario root (default: train4)
  --scenario-name NAME      Scenario name/path filter (repeatable)
  --max-scenarios N         Max scenarios to evaluate (0 = all)
  --out-dir PATH            Output directory for reports + leaderboard
  -h, --help                Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario-root)
      SCENARIO_ROOT="$2"
      shift 2
      ;;
    --scenario-name)
      SCENARIO_NAMES+=("$2")
      shift 2
      ;;
    --max-scenarios)
      MAX_SCENARIOS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

REPORT_DIR="${OUT_DIR}/reports"
mkdir -p "${REPORT_DIR}"

MANIFEST_PATH="${OUT_DIR}/run_manifest.jsonl"
: > "${MANIFEST_PATH}"

JUDGE_BASE=(
  python3 tools/judge_intersections.py
  --scenario-root "${SCENARIO_ROOT}"
  --max-scenarios "${MAX_SCENARIOS}"
  --top-actors-per-scenario 8
  --top-windows-per-track 4
)
for sname in "${SCENARIO_NAMES[@]}"; do
  JUDGE_BASE+=(--scenario-name "${sname}")
done

# Format: name|processing_profile|env_overrides (space-separated NAME=VALUE pairs)
VARIANTS=(
  "baseline|current|"
  "legacy_profile|legacy_stable|"
  "v2_guard|current|V2X_ALIGN_INTERSECTION_BLEND_MAX_LOW_MOTION_MEAN_STEP_M=0.35 V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_RATIO=2.0 V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M=0.55"
  "carla_low_motion_hold|current|V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M=1.4 V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN=1.0 V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE=1.8 V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS_M=0.3"
  "shape_strict|current|V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M=2.6 V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M=2.4 V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M=0.6 V2X_CARLA_INTERSECTION_CLUSTER_MAX_WORSEN_PER_FRAME_M=0.6"
  "combo_guarded|current|V2X_ALIGN_INTERSECTION_BLEND_MAX_LOW_MOTION_MEAN_STEP_M=0.35 V2X_ALIGN_APPROACH_LOCK_STEP_GUARD_RATIO=2.0 V2X_ALIGN_RETIME_TRANSITION_MIN_IMPROVEMENT_M=0.55 V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_DIST_GAIN_M=1.4 V2X_CARLA_LOW_MOTION_OVERRIDE_MIN_SCORE_GAIN=1.0 V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_SCALE=1.8 V2X_CARLA_LOW_MOTION_OVERRIDE_MAX_JUMP_BIAS_M=0.3 V2X_CARLA_INTERSECTION_SHAPE_MAX_QUERY_OFFSET_M=2.6 V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_OFFSET_M=2.4 V2X_CARLA_INTERSECTION_SHAPE_MAX_RAW_WORSEN_PEAK_M=0.6 V2X_CARLA_INTERSECTION_CLUSTER_MAX_WORSEN_PER_FRAME_M=0.6"
)

echo "[INFO] scenario_root=${SCENARIO_ROOT}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] variants=${#VARIANTS[@]}"

for spec in "${VARIANTS[@]}"; do
  IFS='|' read -r variant_name processing_profile env_blob <<< "${spec}"
  report_path="${REPORT_DIR}/${variant_name}.json"
  html_path="${REPORT_DIR}/${variant_name}.html"

  env_cmd=(env)
  if [[ -n "${env_blob}" ]]; then
    # shellcheck disable=SC2206
    env_parts=(${env_blob})
    env_cmd+=("${env_parts[@]}")
  fi

  cmd=(
    "${JUDGE_BASE[@]}"
    --processing-profile "${processing_profile}"
    --output "${report_path}"
    --html-output "${html_path}"
  )

  echo "[RUN] ${variant_name} (profile=${processing_profile})"
  set +e
  "${env_cmd[@]}" "${cmd[@]}"
  rc=$?
  set -e

  python3 - <<PY >> "${MANIFEST_PATH}"
import json
print(json.dumps({
  "variant": "${variant_name}",
  "processing_profile": "${processing_profile}",
  "env_blob": "${env_blob}",
  "report_path": "${report_path}",
  "html_path": "${html_path}",
  "exit_code": int(${rc}),
}))
PY

  python3 - <<PY
import json
from pathlib import Path
p = Path("${report_path}")
if p.exists():
  data = json.loads(p.read_text(encoding="utf-8"))
  s = data.get("summary", {}) if isinstance(data, dict) else {}
  hard_w = int(s.get("hard_fail_windows", 0))
  win_n = int(s.get("intersection_windows", 0))
  bad_sc = int(s.get("scenarios_with_hard_fails", 0))
  print(f"[RESULT] ${variant_name}: hard_fail_windows={hard_w} windows={win_n} scenarios_with_hard_fails={bad_sc} rc=${rc}")
else:
  print(f"[RESULT] ${variant_name}: missing report rc=${rc}")
PY
done

LEADERBOARD_JSON="${OUT_DIR}/leaderboard.json"
LEADERBOARD_HTML="${OUT_DIR}/leaderboard.html"

python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path("${MANIFEST_PATH}")
rows = []
for line in manifest_path.read_text(encoding="utf-8").splitlines():
  line = line.strip()
  if not line:
    continue
  rows.append(json.loads(line))

entries = []
for row in rows:
  report_path = Path(str(row.get("report_path", "")))
  summary = {}
  if report_path.exists():
    try:
      payload = json.loads(report_path.read_text(encoding="utf-8"))
      if isinstance(payload, dict):
        summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
    except Exception:
      summary = {}
  hard_counts = summary.get("hard_fail_type_counts", {}) if isinstance(summary.get("hard_fail_type_counts", {}), dict) else {}
  score_tuple = (
    int(summary.get("hard_fail_windows", 10**9)),
    int(summary.get("scenarios_with_hard_fails", 10**9)),
    int(hard_counts.get("goes_around_intersection", 10**9)),
    int(hard_counts.get("mode_flip", 10**9)),
    int(hard_counts.get("jump_spike", 10**9)),
    int(hard_counts.get("off_raw", 10**9)),
    int(row.get("exit_code", 999)),
  )
  entries.append({
    "variant": str(row.get("variant", "")),
    "processing_profile": str(row.get("processing_profile", "")),
    "env_blob": str(row.get("env_blob", "")),
    "exit_code": int(row.get("exit_code", 0)),
    "report_path": str(report_path),
    "html_path": str(row.get("html_path", "")),
    "summary": summary,
    "hard_fail_type_counts": hard_counts,
    "_score_tuple": score_tuple,
  })

entries.sort(key=lambda x: x["_score_tuple"])
for i, row in enumerate(entries, start=1):
  row["rank"] = int(i)
  row.pop("_score_tuple", None)

leaderboard = {
  "generated_at_utc": datetime.now(timezone.utc).isoformat(),
  "scenario_root": "${SCENARIO_ROOT}",
  "out_dir": "${OUT_DIR}",
  "variant_count": len(entries),
  "entries": entries,
}
Path("${LEADERBOARD_JSON}").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")
print(f"[INFO] Wrote leaderboard JSON: ${LEADERBOARD_JSON}")
PY

python3 - <<PY
import html
import json
from pathlib import Path

data = json.loads(Path("${LEADERBOARD_JSON}").read_text(encoding="utf-8"))
rows = []
for row in data.get("entries", []):
  summary = row.get("summary", {}) if isinstance(row.get("summary", {}), dict) else {}
  fail_counts = row.get("hard_fail_type_counts", {}) if isinstance(row.get("hard_fail_type_counts", {}), dict) else {}
  rows.append(
    "<tr>"
    f"<td>{int(row.get('rank', 0))}</td>"
    f"<td>{html.escape(str(row.get('variant', '')))}</td>"
    f"<td>{html.escape(str(row.get('processing_profile', '')))}</td>"
    f"<td>{int(summary.get('hard_fail_windows', 0))}</td>"
    f"<td>{int(summary.get('scenarios_with_hard_fails', 0))}</td>"
    f"<td>{int(summary.get('intersection_windows', 0))}</td>"
    f"<td>{int(fail_counts.get('goes_around_intersection', 0))}</td>"
    f"<td>{int(fail_counts.get('mode_flip', 0))}</td>"
    f"<td>{int(fail_counts.get('off_raw', 0))}</td>"
    f"<td>{int(fail_counts.get('jump_spike', 0))}</td>"
    f"<td>{int(row.get('exit_code', 0))}</td>"
    "</tr>"
  )

html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Intersection Sweep Leaderboard</title>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 20px; background:#0f1720; color:#e4edf3; }}
    .card {{ background:#142434; border:1px solid #27445f; border-radius:8px; padding:12px; margin-bottom:12px; }}
    table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border:1px solid #2f4f69; padding:6px 8px; text-align:left; }}
    th {{ background:#1b3348; }}
    .mono {{ font-family: monospace; }}
  </style>
</head>
<body>
  <h1>Intersection Sweep Leaderboard</h1>
  <div class="card">
    <div><b>Generated:</b> {html.escape(str(data.get('generated_at_utc', '-')))}</div>
    <div><b>Scenario root:</b> <span class="mono">{html.escape(str(data.get('scenario_root', '-')))}</span></div>
    <div><b>Out dir:</b> <span class="mono">{html.escape(str(data.get('out_dir', '-')))}</span></div>
  </div>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Rank</th><th>Variant</th><th>Profile</th><th>Hard Windows</th>
          <th>Scenarios w/ Fails</th><th>Total Windows</th><th>around</th>
          <th>mode_flip</th><th>off_raw</th><th>jump_spike</th><th>RC</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""

Path("${LEADERBOARD_HTML}").write_text(html_text, encoding="utf-8")
print(f"[INFO] Wrote leaderboard HTML: ${LEADERBOARD_HTML}")
PY

echo "[DONE] Sweep completed."
echo "  JSON: ${LEADERBOARD_JSON}"
echo "  HTML: ${LEADERBOARD_HTML}"
