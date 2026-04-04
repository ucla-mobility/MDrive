#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_carla_rootcause_capture.sh [options] [-- <extra Carla args>]

Options:
  --carla-root PATH   Carla root (default: /data2/marco/CoLMDriver/carla912)
  --logdir PATH       Output directory (default: /data2/marco/CoLMDriver/rootcause_YYYYmmdd_HHMMSS)
  --gpu ID            CUDA_VISIBLE_DEVICES value (default: keep current env)
  --port N            Preferred CARLA RPC port; the wrapper will choose a free port bundle near it (default: 4010)
  --traffic-manager-port N
                      Preferred Traffic Manager port; if omitted, infer from --eval-cmd or use port + 5
  --port-tries N      Search window for a free CARLA/TM port bundle (default: 8)
  --port-step N       Port increment when scanning for a free CARLA/TM port bundle (default: 1)
  --mode MODE         Capture mode: auto, gdb, or core (default: auto)
  --diag-python PATH  Python for carla_server_diag.py (default: auto-pick)
  --diag-interval N   Diagnostics polling interval in seconds (default: 2)
  --startup-timeout N Seconds to wait for CARLA child PID under gdb (default: 180)
  --core-wait-seconds N
                      Seconds to wait for a core file after CARLA exits in core mode (default: 15)
  --stop-after-eval   Stop CARLA/gdb once --eval-cmd finishes instead of waiting for CARLA to exit
  --repeat-eval-until-crash
                      Keep rerunning --eval-cmd until CARLA exits or gdb stops on a crash
  --repeat-eval-delay N
                      Seconds to sleep between repeated eval runs (default: 0)
  --max-eval-runs N   Optional cap for repeated eval runs; 0 means unlimited (default: 0)
  --eval-cmd CMD      Optional eval command to run and tee to eval.log
  --help              Show this message

Examples:
  tools/run_carla_rootcause_capture.sh --gpu 6 --port 4010 -- \
    -RenderOffScreen

  tools/run_carla_rootcause_capture.sh --gpu 6 --port 4010 \
    --eval-cmd "cd /data2/marco/CoLMDriver && python tools/run_custom_eval.py ..."
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTCAUSE_HELPER="$SCRIPT_DIR/carla_rootcause_helper.py"

CARLA_ROOT="/data2/marco/CoLMDriver/carla912"
LOGDIR=""
GPU_VALUE="${CUDA_VISIBLE_DEVICES:-}"
HOST="127.0.0.1"
PORT=4010
REQUESTED_PORT=""
TM_PORT=""
TM_PORT_OFFSET=""
PORT_TRIES=8
PORT_STEP=1
STREAM_PORT_OFFSET=1
RUN_MODE="auto"
DIAG_PY_OVERRIDE="${CARLA_DIAG_PYTHON:-}"
DIAG_INTERVAL=2
STARTUP_TIMEOUT_SEC=180
CORE_WAIT_SECONDS=15
STOP_AFTER_EVAL=0
REPEAT_EVAL_UNTIL_CRASH=0
REPEAT_EVAL_DELAY_SECONDS=0
MAX_EVAL_RUNS=0
EVAL_CMD=""
EVAL_CMD_RESOLVED=""
EVAL_USES_RUN_CUSTOM_EVAL=0
EVAL_HAS_START_CARLA=0
EVAL_INFERRED_PORT=""
EVAL_INFERRED_TM_PORT=""
EVAL_INFERRED_TM_PORT_OFFSET=""
CARLA_EXTRA_ARGS=()

CARLA_BIN=""
DIAG_SCRIPT=""
CARLA_RUNTIME_LOG=""
CARLA_CONSOLE_LOG=""
GDB_CMD_FILE=""
POSTMORTEM_GDB_CMD_FILE=""
POSTMORTEM_GDB_LOG=""

GDB_PID=""
DIAG_PID=""
CARLA_PID=""
TRACE_MODE_USED=""
CARLA_EXIT_CODE=0
GDB_EXIT_CODE=0
EVAL_EXIT_CODE=0
HELPER_RUNNER=""
LAST_ERROR=""
CORE_FILE=""
INTENTIONAL_STOP=0
EVAL_RUN_COUNT=0
EVAL_LOOP_END_REASON=""
PORT_SELECTION_REASON=""
PORT_SELECTION_CLEANUP_CLOSED=0
PORT_SELECTION_UNMATCHED_COUNT=0
SELECTED_SERVICE_PORTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --carla-root)
      CARLA_ROOT="$2"
      shift 2
      ;;
    --logdir)
      LOGDIR="$2"
      shift 2
      ;;
    --gpu)
      GPU_VALUE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --traffic-manager-port)
      TM_PORT="$2"
      shift 2
      ;;
    --port-tries)
      PORT_TRIES="$2"
      shift 2
      ;;
    --port-step)
      PORT_STEP="$2"
      shift 2
      ;;
    --mode)
      RUN_MODE="$2"
      shift 2
      ;;
    --diag-python)
      DIAG_PY_OVERRIDE="$2"
      shift 2
      ;;
    --diag-interval)
      DIAG_INTERVAL="$2"
      shift 2
      ;;
    --startup-timeout)
      STARTUP_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --core-wait-seconds)
      CORE_WAIT_SECONDS="$2"
      shift 2
      ;;
    --stop-after-eval)
      STOP_AFTER_EVAL=1
      shift
      ;;
    --repeat-eval-until-crash)
      REPEAT_EVAL_UNTIL_CRASH=1
      shift
      ;;
    --repeat-eval-delay)
      REPEAT_EVAL_DELAY_SECONDS="$2"
      shift 2
      ;;
    --max-eval-runs)
      MAX_EVAL_RUNS="$2"
      shift 2
      ;;
    --eval-cmd)
      EVAL_CMD="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      CARLA_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LOGDIR" ]]; then
  LOGDIR="/data2/marco/CoLMDriver/rootcause_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOGDIR"
REQUESTED_PORT="$PORT"

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --port must be an integer (got: $PORT)" >&2
  exit 2
fi
if [[ -n "$TM_PORT" ]] && ! [[ "$TM_PORT" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --traffic-manager-port must be an integer (got: $TM_PORT)" >&2
  exit 2
fi
if ! [[ "$PORT_TRIES" =~ ^[0-9]+$ ]] || (( PORT_TRIES <= 0 )); then
  echo "[ERROR] --port-tries must be a positive integer (got: $PORT_TRIES)" >&2
  exit 2
fi
if ! [[ "$PORT_STEP" =~ ^[0-9]+$ ]] || (( PORT_STEP <= 0 )); then
  echo "[ERROR] --port-step must be a positive integer (got: $PORT_STEP)" >&2
  exit 2
fi
case "$RUN_MODE" in
  auto|gdb|core)
    ;;
  *)
    echo "[ERROR] --mode must be one of: auto, gdb, core (got: $RUN_MODE)" >&2
    exit 2
    ;;
esac
if ! [[ "$DIAG_INTERVAL" =~ ^[0-9]+$ ]] || (( DIAG_INTERVAL <= 0 )); then
  echo "[ERROR] --diag-interval must be a positive integer (got: $DIAG_INTERVAL)" >&2
  exit 2
fi
if ! [[ "$STARTUP_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || (( STARTUP_TIMEOUT_SEC <= 0 )); then
  echo "[ERROR] --startup-timeout must be a positive integer (got: $STARTUP_TIMEOUT_SEC)" >&2
  exit 2
fi
if ! [[ "$CORE_WAIT_SECONDS" =~ ^[0-9]+$ ]] || (( CORE_WAIT_SECONDS < 0 )); then
  echo "[ERROR] --core-wait-seconds must be a non-negative integer (got: $CORE_WAIT_SECONDS)" >&2
  exit 2
fi
if ! [[ "$REPEAT_EVAL_DELAY_SECONDS" =~ ^[0-9]+$ ]] || (( REPEAT_EVAL_DELAY_SECONDS < 0 )); then
  echo "[ERROR] --repeat-eval-delay must be a non-negative integer (got: $REPEAT_EVAL_DELAY_SECONDS)" >&2
  exit 2
fi
if ! [[ "$MAX_EVAL_RUNS" =~ ^[0-9]+$ ]] || (( MAX_EVAL_RUNS < 0 )); then
  echo "[ERROR] --max-eval-runs must be a non-negative integer (got: $MAX_EVAL_RUNS)" >&2
  exit 2
fi

CARLA_BIN="$CARLA_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
DIAG_SCRIPT="$CARLA_ROOT/PythonAPI/util/carla_server_diag.py"
CARLA_RUNTIME_LOG="$LOGDIR/carla_runtime.log"
CARLA_CONSOLE_LOG="$LOGDIR/carla_console.log"
GDB_CMD_FILE="$LOGDIR/gdb_commands.gdb"
POSTMORTEM_GDB_CMD_FILE="$LOGDIR/postmortem_gdb_commands.gdb"
POSTMORTEM_GDB_LOG="$LOGDIR/postmortem_gdb.log"

if [[ ! -x "$CARLA_BIN" ]]; then
  echo "[ERROR] CARLA binary not found/executable: $CARLA_BIN" >&2
  exit 1
fi
if [[ ! -f "$DIAG_SCRIPT" ]]; then
  echo "[ERROR] Diagnostic script not found: $DIAG_SCRIPT" >&2
  exit 1
fi
if ! command -v gdb >/dev/null 2>&1; then
  if [[ "$RUN_MODE" == "gdb" ]]; then
    echo "[ERROR] gdb not found in PATH." >&2
    exit 1
  fi
  echo "[WARN] gdb not found in PATH; postmortem stack traces will be unavailable." >&2
fi

pick_diag_python() {
  local candidates=()
  if [[ -n "$DIAG_PY_OVERRIDE" ]]; then
    candidates+=("$DIAG_PY_OVERRIDE")
  fi
  candidates+=(
    "/data/miniconda3/envs/b2d_zoo/bin/python"
    "/data/miniconda3/envs/run_custom_eval_baseline/bin/python"
    "python3"
    "python"
  )

  local py=""
  for py in "${candidates[@]}"; do
    if ! command -v "$py" >/dev/null 2>&1; then
      continue
    fi
    if "$py" - <<'PY' >/dev/null 2>&1
import carla  # noqa: F401
PY
    then
      echo "$py"
      return 0
    fi
  done

  for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "$py"
      return 0
    fi
  done
  return 1
}

reset_logdir_outputs() {
  local f=""
  for f in \
    run_meta.txt \
    host_snapshot.txt \
    env_snapshot.txt \
    process_snapshot_start.txt \
    process_snapshot_end.txt \
    port_snapshot_selected.txt \
    port_snapshot_start.txt \
    port_snapshot_end.txt \
    gdb_commands.gdb \
    gdb_console.log \
    gdb_internal.log \
    carla_runtime.log \
    carla_console.log \
    carla_proc_start.txt \
    carla_proc_end.txt \
    nvidia_smi_start.txt \
    nvidia_smi_end.txt \
    diag.log \
    postmortem_gdb_commands.gdb \
    postmortem_gdb.log \
    eval.log \
    combined.log; do
    rm -f "$LOGDIR/$f"
  done
}

inspect_eval_command() {
  local key=""
  local value=""
  local helper_output=""

  EVAL_USES_RUN_CUSTOM_EVAL=0
  EVAL_HAS_START_CARLA=0
  EVAL_INFERRED_PORT=""
  EVAL_INFERRED_TM_PORT=""
  EVAL_INFERRED_TM_PORT_OFFSET=""

  [[ -n "$EVAL_CMD" ]] || return 0

  if ! helper_output="$("$HELPER_RUNNER" "$ROOTCAUSE_HELPER" inspect-eval-cmd --eval-cmd "$EVAL_CMD")"; then
    echo "[ERROR] Failed to inspect --eval-cmd with $ROOTCAUSE_HELPER." >&2
    exit 1
  fi

  while IFS='=' read -r key value; do
    case "$key" in
      uses_run_custom_eval)
        EVAL_USES_RUN_CUSTOM_EVAL="$value"
        ;;
      has_start_carla)
        EVAL_HAS_START_CARLA="$value"
        ;;
      eval_port)
        EVAL_INFERRED_PORT="$value"
        ;;
      eval_tm_port)
        EVAL_INFERRED_TM_PORT="$value"
        ;;
      eval_tm_port_offset)
        EVAL_INFERRED_TM_PORT_OFFSET="$value"
        ;;
    esac
  done <<<"$helper_output"
}

determine_tm_port_offset() {
  if [[ -n "$TM_PORT" ]]; then
    TM_PORT_OFFSET=$((TM_PORT - REQUESTED_PORT))
    if (( TM_PORT_OFFSET == 0 )); then
      echo "[ERROR] --traffic-manager-port must differ from --port." >&2
      exit 2
    fi
    return 0
  fi

  if [[ "$EVAL_USES_RUN_CUSTOM_EVAL" == "1" && -n "$EVAL_INFERRED_TM_PORT_OFFSET" ]]; then
    TM_PORT_OFFSET="$EVAL_INFERRED_TM_PORT_OFFSET"
    return 0
  fi

  TM_PORT_OFFSET=5
}

select_effective_port_bundle() {
  local key=""
  local value=""
  local helper_output=""

  if ! helper_output="$(
    "$HELPER_RUNNER" "$ROOTCAUSE_HELPER" select-port-bundle \
      --host "$HOST" \
      --preferred-port "$REQUESTED_PORT" \
      --port-tries "$PORT_TRIES" \
      --port-step "$PORT_STEP" \
      --tm-port-offset "$TM_PORT_OFFSET" \
      --stream-port-offset "$STREAM_PORT_OFFSET" \
      --cleanup-stale
  )"; then
    echo "[ERROR] Failed to select a CARLA/TM port bundle with $ROOTCAUSE_HELPER." >&2
    exit 1
  fi

  while IFS='=' read -r key value; do
    case "$key" in
      selected_port)
        PORT="$value"
        ;;
      selected_tm_port)
        TM_PORT="$value"
        ;;
      selected_service_ports)
        SELECTED_SERVICE_PORTS="$value"
        ;;
      selection_reason)
        PORT_SELECTION_REASON="$value"
        ;;
      cleanup_closed)
        PORT_SELECTION_CLEANUP_CLOSED="$value"
        ;;
      unmatched_count)
        PORT_SELECTION_UNMATCHED_COUNT="$value"
        ;;
    esac
  done <<<"$helper_output"
}

prepare_eval_command() {
  EVAL_CMD_RESOLVED="$EVAL_CMD"

  [[ -n "$EVAL_CMD" ]] || return 0

  if [[ "$EVAL_USES_RUN_CUSTOM_EVAL" == "1" ]]; then
    if [[ "$EVAL_HAS_START_CARLA" == "1" ]]; then
      echo "[ERROR] --eval-cmd contains run_custom_eval.py with --start-carla, but this wrapper already owns the CARLA process." >&2
      exit 2
    fi
    EVAL_CMD_RESOLVED="$EVAL_CMD_RESOLVED --port $PORT --traffic-manager-port $TM_PORT"
  fi
}

has_carla_arg() {
  local key="$1"
  local arg=""
  for arg in "${CARLA_EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "$key" || "$arg" == "$key="* ]]; then
      return 0
    fi
  done
  return 1
}

log_note() {
  local message="$1"
  echo "$message"
  echo "$message" >>"$LOGDIR/run_meta.txt"
}

set_last_error() {
  LAST_ERROR="$1"
  echo "[ERROR] $LAST_ERROR" >>"$LOGDIR/run_meta.txt"
}

process_exe() {
  local pid="$1"
  readlink -f "/proc/$pid/exe" 2>/dev/null || true
}

process_cmdline() {
  local pid="$1"
  tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true
}

pid_is_alive() {
  local pid="$1"
  [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

pid_matches_exe() {
  local pid="$1"
  local expected="$2"
  [[ "$(process_exe "$pid")" == "$expected" ]]
}

listening_pids_for_port() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  lsof -nP -t "-iTCP:$port" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++'
}

pid_owns_port_listener() {
  local pid="$1"
  local port="$2"
  local owner=""
  while IFS= read -r owner; do
    [[ -n "$owner" ]] || continue
    if [[ "$owner" == "$pid" ]]; then
      return 0
    fi
  done < <(listening_pids_for_port "$port")
  return 1
}

describe_port_listeners_inline() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof-unavailable"
    return 0
  fi
  local details=""
  details="$(lsof -nP "-iTCP:$port" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
  if [[ -z "$details" ]]; then
    echo "none"
  else
    echo "$details"
  fi
}

wait_for_pid_to_own_port() {
  local pid="$1"
  local port="$2"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    if ! pid_is_alive "$pid"; then
      return 1
    fi
    if pid_owns_port_listener "$pid" "$port"; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

capture_host_snapshot() {
  {
    echo "timestamp=$(date -Iseconds)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo
    echo "==== ulimit -a ===="
    ulimit -a
    echo
    echo "==== /proc/sys/fs ===="
    for f in /proc/sys/fs/file-max /proc/sys/fs/file-nr /proc/sys/fs/nr_open; do
      if [[ -f "$f" ]]; then
        echo "$f=$(cat "$f")"
      fi
    done
    echo
    echo "==== /proc/sys/kernel ===="
    for f in /proc/sys/kernel/core_pattern /proc/sys/kernel/core_uses_pid /proc/sys/kernel/yama/ptrace_scope; do
      if [[ -f "$f" ]]; then
        echo "$f=$(cat "$f")"
      fi
    done
    echo
    echo "==== nvidia-smi -L ===="
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L
    else
      echo "nvidia-smi not found"
    fi
  } >"$LOGDIR/host_snapshot.txt" 2>&1 || true

  env | sort >"$LOGDIR/env_snapshot.txt" 2>&1 || true
}

capture_process_list() {
  local outfile="$1"
  ps -eo pid,ppid,pgid,sid,stat,etime,cmd \
    | awk 'NR==1 || /CarlaUE4-Linux-Shipping|carla_server_diag\.py|[[:space:]]gdb([[:space:]]|$)/' \
    >"$outfile" 2>/dev/null || true
}

capture_port_snapshot() {
  local outfile="$1"
  local seen=""
  local candidate=""

  {
    echo "timestamp=$(date -Iseconds)"
    echo "host=$HOST"
    echo "rpc_port=$PORT"
    echo "tm_port=${TM_PORT:-}"
    echo "stream_port=$((PORT + STREAM_PORT_OFFSET))"
    echo "service_ports=${SELECTED_SERVICE_PORTS:-$PORT,$((PORT + STREAM_PORT_OFFSET))${TM_PORT:+,$TM_PORT}}"
    for candidate in "$PORT" "$((PORT + STREAM_PORT_OFFSET))" "${TM_PORT:-}"; do
      [[ -n "$candidate" ]] || continue
      if [[ ",$seen," == *",$candidate,"* ]]; then
        continue
      fi
      seen="${seen:+$seen,}$candidate"
      echo
      echo "==== TCP LISTEN $candidate ===="
      if command -v lsof >/dev/null 2>&1; then
        lsof -nP "-iTCP:$candidate" -sTCP:LISTEN 2>/dev/null || true
      else
        echo "lsof not found"
      fi
    done
  } >"$outfile" 2>&1 || true
}

capture_proc_snapshot() {
  local pid="$1"
  local outfile="$2"
  {
    echo "timestamp=$(date -Iseconds)"
    echo "pid=$pid"

    if [[ ! -d "/proc/$pid" ]]; then
      echo "process_alive=0"
    else
      echo "process_alive=1"
      echo "exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
      echo "cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
      echo
      echo "==== /proc/$pid/status ===="
      cat "/proc/$pid/status"
      echo
      echo "==== /proc/$pid/limits ===="
      cat "/proc/$pid/limits"
      echo
      echo "==== /proc/$pid/io ===="
      cat "/proc/$pid/io" 2>/dev/null || true
      echo
      echo "==== /proc/$pid/fd count ===="
      ls -1 "/proc/$pid/fd" 2>/dev/null | wc -l
      echo
      echo "==== /proc/$pid/fd list ===="
      ls -l "/proc/$pid/fd" 2>/dev/null || true
    fi
  } >"$outfile" 2>&1 || true
}

get_ppid() {
  local pid="$1"
  awk '/^PPid:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true
}

pid_has_ancestor() {
  local pid="$1"
  local ancestor="$2"
  local hops=0
  local ppid=""
  while [[ -n "$pid" && "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 && $hops -lt 128 ]]; do
    if [[ "$pid" == "$ancestor" ]]; then
      return 0
    fi
    ppid="$(get_ppid "$pid")"
    if [[ -z "$ppid" ]]; then
      break
    fi
    pid="$ppid"
    ((hops++))
  done
  return 1
}

find_carla_pid_under_gdb() {
  local gdb_pid="$1"
  local proc_dir=""
  local pid=""
  local exe=""
  local newest=""

  for proc_dir in /proc/[0-9]*; do
    pid="${proc_dir##*/}"
    [[ "$pid" == "$gdb_pid" ]] && continue
    exe="$(readlink -f "$proc_dir/exe" 2>/dev/null || true)"
    [[ "$exe" == "$CARLA_BIN" ]] || continue

    if pid_has_ancestor "$pid" "$gdb_pid"; then
      if [[ -z "$newest" || "$pid" -gt "$newest" ]]; then
        newest="$pid"
      fi
    fi
  done

  if [[ -n "$newest" ]]; then
    echo "$newest"
    return 0
  fi
  return 1
}

find_latest_core_file() {
  local latest=""
  local candidate=""

  if [[ -n "$CARLA_PID" ]]; then
    for candidate in "$CARLA_ROOT/core.$CARLA_PID" "$CARLA_ROOT/core"; do
      if [[ -f "$candidate" ]]; then
        latest="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$latest" ]]; then
    latest="$(
      find "$CARLA_ROOT" -maxdepth 1 -type f -name 'core*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | awk 'NR==1 {print $2}'
    )"
  fi

  if [[ -n "$latest" && -f "$latest" ]]; then
    echo "$latest"
    return 0
  fi
  return 1
}

start_diag_sidecar() {
  LAST_ERROR=""
  if ! pid_is_alive "$CARLA_PID"; then
    set_last_error "Cannot start diagnostics; CARLA pid '$CARLA_PID' is not alive."
    return 1
  fi
  if ! pid_matches_exe "$CARLA_PID" "$CARLA_BIN"; then
    set_last_error \
      "Diagnostics target pid '$CARLA_PID' is not CARLA. exe=$(process_exe "$CARLA_PID") cmdline=$(process_cmdline "$CARLA_PID")"
    return 1
  fi
  if ! pid_owns_port_listener "$CARLA_PID" "$PORT"; then
    set_last_error \
      "Diagnostics target pid '$CARLA_PID' does not own RPC port $PORT. listeners=$(describe_port_listeners_inline "$PORT")"
    return 1
  fi

  log_note "[ROOTCAUSE] starting diagnostics sidecar for pid=$CARLA_PID..."
  (
    "$DIAG_PY" "$DIAG_SCRIPT" \
      --pid "$CARLA_PID" \
      --host "$HOST" \
      --port "$PORT" \
      --interval "$DIAG_INTERVAL" \
      --connect-timeout 2 \
      --project-root "$CARLA_ROOT" \
      --top-k 120 \
      --actor-event-limit 10000 \
      --sensor-detail-limit 10000 \
      --fd-detail-limit 40000 \
      --fd-top-k 400 \
      --peer-top-k 400 \
      --thread-top-k 200 \
      --socket-refresh-every 1 \
      --gpu-refresh-every 1 \
      --dump-all-actors \
      --dump-all-fds \
      --dump-all-sensors \
      --alert-fd-high 1 \
      --alert-fd-jump 1 \
      --alert-nvidia-fd-high 1 \
      --alert-nvidia-fd-jump 1 \
      --alert-rss-jump-mb 64 \
      --alert-vmsize-jump-mb 64 \
      --alert-gpu-mem-jump-mb 64 \
      --actor-chunk-size 200 \
      --fd-chunk-size 300 \
      --vma-scan-every 1 \
      --vma-top-k 80 \
      --fd-delta-top-k 200 \
      --stack-cooldown-s 10 \
      --enable-stack-snapshot 1 \
      |& tee -a "$LOGDIR/diag.log"
  ) &
  DIAG_PID=$!
  echo "diag_pid=$DIAG_PID" >>"$LOGDIR/run_meta.txt"
  return 0
}

generate_postmortem_gdb_cmd_file() {
  cat >"$POSTMORTEM_GDB_CMD_FILE" <<'EOF_GDB_POST'
set pagination off
set confirm off
set print thread-events off
set backtrace limit 128
echo \n===== POSTMORTEM_GDB_CONTEXT =====\n
info program
bt full
thread apply all bt 32
info threads
info registers
x/32i $pc-64
info proc mappings
info sharedlibrary
quit
EOF_GDB_POST
}

run_postmortem_gdb() {
  local core_file="$1"
  if ! command -v gdb >/dev/null 2>&1; then
    log_note "[WARN] gdb unavailable; skipping postmortem analysis for core $core_file."
    return 0
  fi
  generate_postmortem_gdb_cmd_file
  log_note "[ROOTCAUSE] running postmortem gdb on core: $core_file"
  gdb -q -x "$POSTMORTEM_GDB_CMD_FILE" "$CARLA_BIN" "$core_file" \
    >"$POSTMORTEM_GDB_LOG" 2>&1 || true
}

wait_for_core_file() {
  local deadline=$((SECONDS + CORE_WAIT_SECONDS))
  local core_file=""

  while :; do
    core_file="$(find_latest_core_file || true)"
    if [[ -n "$core_file" ]]; then
      echo "$core_file"
      return 0
    fi
    if (( SECONDS >= deadline )); then
      break
    fi
    sleep 1
  done
  return 1
}

wait_for_pid_exit() {
  local pid="$1"
  local timeout_s="${2:-5}"
  local deadline=$((SECONDS + timeout_s))

  while pid_is_alive "$pid"; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep 0.2
  done
  return 0
}

gdb_has_stop_context() {
  [[ -f "$LOGDIR/gdb_console.log" ]] && grep -q "===== GDB_STOP_CONTEXT =====" "$LOGDIR/gdb_console.log"
}

capture_target_has_stopped() {
  if [[ "$TRACE_MODE_USED" == "gdb" ]]; then
    if ! pid_is_alive "${GDB_PID:-}"; then
      return 0
    fi
    if gdb_has_stop_context; then
      return 0
    fi
    if ! pid_is_alive "${CARLA_PID:-}"; then
      return 0
    fi
    return 1
  fi

  if ! pid_is_alive "${CARLA_PID:-}"; then
    return 0
  fi
  return 1
}

stop_active_target() {
  if pid_is_alive "${CARLA_PID:-}"; then
    INTENTIONAL_STOP=1
    log_note "[ROOTCAUSE] stopping CARLA pid=$CARLA_PID after eval finished."
    kill "$CARLA_PID" >/dev/null 2>&1 || true
    if ! wait_for_pid_exit "$CARLA_PID" 5; then
      log_note "[WARN] CARLA pid=$CARLA_PID did not exit after SIGTERM; sending SIGKILL."
      kill -KILL "$CARLA_PID" >/dev/null 2>&1 || true
    fi
  fi
  if [[ "$TRACE_MODE_USED" == "gdb" ]] && pid_is_alive "${GDB_PID:-}"; then
    INTENTIONAL_STOP=1
    log_note "[ROOTCAUSE] stopping gdb pid=$GDB_PID after eval finished."
    kill -TERM "$GDB_PID" >/dev/null 2>&1 || true
    if ! wait_for_pid_exit "$GDB_PID" 5; then
      log_note "[WARN] gdb pid=$GDB_PID did not exit after SIGTERM; sending SIGKILL."
      kill -KILL "$GDB_PID" >/dev/null 2>&1 || true
    fi
  fi
  return 0
}

run_eval_once() {
  local run_id=$((EVAL_RUN_COUNT + 1))
  log_note "[ROOTCAUSE] running eval command (run $run_id)..."
  set +e
  env \
    PYTHONUNBUFFERED=1 \
    CARLA_ROOTCAUSE_HOST="$HOST" \
    CARLA_ROOTCAUSE_PORT="$PORT" \
    CARLA_ROOTCAUSE_TM_PORT="$TM_PORT" \
    CARLA_ROOTCAUSE_CARLA_PID="${CARLA_PID:-}" \
    CARLA_ROOTCAUSE_TRACE_MODE="${TRACE_MODE_USED:-}" \
    CARLA_ROOTCAUSE_LOGDIR="$LOGDIR" \
    bash -lc "$EVAL_CMD_RESOLVED" |& tee -a "$LOGDIR/eval.log"
  EVAL_EXIT_CODE="${PIPESTATUS[0]}"
  set -e
  EVAL_RUN_COUNT="$run_id"
  echo "eval_run_count=$EVAL_RUN_COUNT" >>"$LOGDIR/run_meta.txt"
  echo "eval_run_${run_id}_exit_code=$EVAL_EXIT_CODE" >>"$LOGDIR/run_meta.txt"
  echo "eval_exit_code=$EVAL_EXIT_CODE" >>"$LOGDIR/run_meta.txt"
  if (( EVAL_EXIT_CODE != 0 )); then
    log_note "[WARN] Eval command run $run_id exited with code $EVAL_EXIT_CODE."
  fi
}

launch_carla_under_gdb() {
  LAST_ERROR=""
  if ! command -v gdb >/dev/null 2>&1; then
    set_last_error "gdb is unavailable."
    return 1
  fi

  log_note "[ROOTCAUSE] starting gdb..."
  # Inject close() EBADF shim to suppress the double-close race in clmdep_asio
  local _ebadf_so="$SCRIPT_DIR/close_ebadf_suppress.so"
  local -a gdb_cmd=(gdb -q -x "$GDB_CMD_FILE" --args "$CARLA_BIN" "CarlaUE4" "-carla-rpc-port=$PORT" "${CARLA_EXTRA_ARGS[@]}")
  if [[ -n "$GPU_VALUE" ]]; then
    (
      cd "$CARLA_ROOT"
      env CUDA_VISIBLE_DEVICES="$GPU_VALUE" LD_PRELOAD="${_ebadf_so}${LD_PRELOAD:+:$LD_PRELOAD}" "${gdb_cmd[@]}"
    ) >"$LOGDIR/gdb_console.log" 2>&1 &
  else
    (
      cd "$CARLA_ROOT"
      env LD_PRELOAD="${_ebadf_so}${LD_PRELOAD:+:$LD_PRELOAD}" "${gdb_cmd[@]}"
    ) >"$LOGDIR/gdb_console.log" 2>&1 &
  fi
  GDB_PID=$!
  echo "gdb_pid=$GDB_PID" >>"$LOGDIR/run_meta.txt"

  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    if ! pid_is_alive "$GDB_PID"; then
      break
    fi
    CARLA_PID="$(find_carla_pid_under_gdb "$GDB_PID" || true)"
    if [[ -n "$CARLA_PID" ]] && pid_matches_exe "$CARLA_PID" "$CARLA_BIN"; then
      if wait_for_pid_to_own_port "$CARLA_PID" "$PORT" "$STARTUP_TIMEOUT_SEC"; then
        TRACE_MODE_USED="gdb"
        echo "trace_mode_used=$TRACE_MODE_USED" >>"$LOGDIR/run_meta.txt"
        echo "carla_pid=$CARLA_PID" >>"$LOGDIR/run_meta.txt"
        capture_proc_snapshot "$CARLA_PID" "$LOGDIR/carla_proc_start.txt"
        capture_port_snapshot "$LOGDIR/port_snapshot_start.txt"
        return 0
      fi
      break
    fi
    sleep 0.2
  done

  local gdb_excerpt=""
  if [[ -f "$LOGDIR/gdb_console.log" ]]; then
    gdb_excerpt="$(tail -n 20 "$LOGDIR/gdb_console.log" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
  fi
  set_last_error \
    "Could not confirm that CARLA child pid '${CARLA_PID:-<unknown>}' owned RPC port $PORT under gdb pid '$GDB_PID' within ${STARTUP_TIMEOUT_SEC}s. listeners=$(describe_port_listeners_inline "$PORT"). ${gdb_excerpt}"
  return 1
}

launch_carla_direct() {
  LAST_ERROR=""
  log_note "[ROOTCAUSE] starting CARLA directly for core/postmortem capture..."
  # Inject close() EBADF shim to suppress the double-close race in clmdep_asio
  local _ebadf_so="$SCRIPT_DIR/close_ebadf_suppress.so"
  if [[ -n "$GPU_VALUE" ]]; then
    (
      cd "$CARLA_ROOT"
      env CUDA_VISIBLE_DEVICES="$GPU_VALUE" LD_PRELOAD="${_ebadf_so}${LD_PRELOAD:+:$LD_PRELOAD}" "$CARLA_BIN" "CarlaUE4" "-carla-rpc-port=$PORT" "${CARLA_EXTRA_ARGS[@]}"
    ) >"$CARLA_CONSOLE_LOG" 2>&1 &
  else
    (
      cd "$CARLA_ROOT"
      env LD_PRELOAD="${_ebadf_so}${LD_PRELOAD:+:$LD_PRELOAD}" "$CARLA_BIN" "CarlaUE4" "-carla-rpc-port=$PORT" "${CARLA_EXTRA_ARGS[@]}"
    ) >"$CARLA_CONSOLE_LOG" 2>&1 &
  fi
  CARLA_PID=$!
  TRACE_MODE_USED="core"
  echo "trace_mode_used=$TRACE_MODE_USED" >>"$LOGDIR/run_meta.txt"
  echo "carla_pid=$CARLA_PID" >>"$LOGDIR/run_meta.txt"

  local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    if ! pid_is_alive "$CARLA_PID"; then
      break
    fi
    if pid_matches_exe "$CARLA_PID" "$CARLA_BIN"; then
      if wait_for_pid_to_own_port "$CARLA_PID" "$PORT" "$STARTUP_TIMEOUT_SEC"; then
        capture_proc_snapshot "$CARLA_PID" "$LOGDIR/carla_proc_start.txt"
        capture_port_snapshot "$LOGDIR/port_snapshot_start.txt"
        return 0
      fi
      break
    fi
    sleep 0.2
  done

  local console_excerpt=""
  if [[ -f "$CARLA_CONSOLE_LOG" ]]; then
    console_excerpt="$(tail -n 20 "$CARLA_CONSOLE_LOG" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
  fi
  set_last_error \
    "CARLA pid '$CARLA_PID' did not own RPC port $PORT within ${STARTUP_TIMEOUT_SEC}s. listeners=$(describe_port_listeners_inline "$PORT"). ${console_excerpt}"
  return 1
}

generate_combined_log() {
  local combined="$LOGDIR/combined.log"
  local f=""
  : >"$combined"

  for f in \
    run_meta.txt \
    host_snapshot.txt \
    env_snapshot.txt \
    process_snapshot_start.txt \
    process_snapshot_end.txt \
    port_snapshot_selected.txt \
    port_snapshot_start.txt \
    port_snapshot_end.txt \
    gdb_commands.gdb \
    gdb_console.log \
    gdb_internal.log \
    carla_runtime.log \
    carla_console.log \
    carla_proc_start.txt \
    carla_proc_end.txt \
    nvidia_smi_start.txt \
    nvidia_smi_end.txt \
    diag.log \
    postmortem_gdb_commands.gdb \
    postmortem_gdb.log \
    eval.log; do
    if [[ -f "$LOGDIR/$f" ]]; then
      {
        echo "===== $f ====="
        cat "$LOGDIR/$f"
        echo
      } >>"$combined" 2>/dev/null || true
    fi
  done
}

cleanup() {
  local rc=$?

  if pid_is_alive "${DIAG_PID:-}"; then
    kill "$DIAG_PID" >/dev/null 2>&1 || true
    wait "$DIAG_PID" >/dev/null 2>&1 || true
  fi

  if pid_is_alive "${GDB_PID:-}"; then
    kill -INT "$GDB_PID" >/dev/null 2>&1 || true
    wait "$GDB_PID" >/dev/null 2>&1 || true
  fi

  if [[ -z "${GDB_PID:-}" ]] && pid_is_alive "${CARLA_PID:-}"; then
    kill "$CARLA_PID" >/dev/null 2>&1 || true
    wait "$CARLA_PID" >/dev/null 2>&1 || true
  fi

  if [[ -n "${CARLA_PID:-}" ]]; then
    capture_proc_snapshot "$CARLA_PID" "$LOGDIR/carla_proc_end.txt"
  fi

  capture_process_list "$LOGDIR/process_snapshot_end.txt"
  capture_port_snapshot "$LOGDIR/port_snapshot_end.txt"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >"$LOGDIR/nvidia_smi_end.txt" 2>&1 || true
  fi

  {
    echo "finished=$(date -Iseconds)"
    echo "exit_code=$rc"
  } >>"$LOGDIR/run_meta.txt"

  generate_combined_log
}

reset_logdir_outputs

DIAG_PY="$(pick_diag_python || true)"
if [[ -z "$DIAG_PY" ]]; then
  echo "[ERROR] Unable to find usable python for diagnostics." >&2
  exit 1
fi
HELPER_RUNNER="$DIAG_PY"

if [[ ! -f "$ROOTCAUSE_HELPER" ]]; then
  echo "[ERROR] Root-cause helper not found: $ROOTCAUSE_HELPER" >&2
  exit 1
fi

inspect_eval_command
determine_tm_port_offset
select_effective_port_bundle
prepare_eval_command

if (( REPEAT_EVAL_UNTIL_CRASH == 1 && STOP_AFTER_EVAL == 1 )); then
  echo "[WARN] --stop-after-eval is incompatible with --repeat-eval-until-crash; disabling stop-after-eval." >&2
  STOP_AFTER_EVAL=0
fi

if [[ ${#CARLA_EXTRA_ARGS[@]} -eq 0 ]]; then
  CARLA_EXTRA_ARGS=(-RenderOffScreen)
fi
if ! has_carla_arg "-log"; then
  CARLA_EXTRA_ARGS+=(-log)
fi
if ! has_carla_arg "-abslog"; then
  CARLA_EXTRA_ARGS+=("-abslog=$CARLA_RUNTIME_LOG")
fi

if ! ulimit -c unlimited 2>/dev/null; then
  echo "[WARN] Could not set 'ulimit -c unlimited'; core dumps may be disabled." >&2
fi

cat >"$GDB_CMD_FILE" <<EOF_GDB
set pagination off
set confirm off
set print thread-events off
set env LD_PRELOAD $SCRIPT_DIR/close_ebadf_suppress.so
set follow-fork-mode parent
set detach-on-fork on
set follow-exec-mode same
set logging file $LOGDIR/gdb_internal.log
set logging overwrite on
set logging redirect off
set logging enabled on
set backtrace limit 128
handle SIGCHLD nostop noprint pass
handle SIGPIPE nostop noprint pass
handle SIGSEGV stop print nopass
handle SIGABRT stop print nopass
handle SIGBUS stop print nopass
handle SIGILL stop print nopass
run
echo \n===== GDB_STOP_CONTEXT =====\n
info program
bt full
thread apply all bt 32
info threads
info registers
x/32i \$pc-64
info proc mappings
info sharedlibrary
quit
EOF_GDB

{
  echo "timestamp=$(date -Iseconds)"
  echo "carla_root=$CARLA_ROOT"
  echo "carla_bin=$CARLA_BIN"
  echo "diag_script=$DIAG_SCRIPT"
  echo "diag_python=$DIAG_PY"
  echo "helper_python=$HELPER_RUNNER"
  echo "host=$HOST"
  echo "requested_port=$REQUESTED_PORT"
  echo "port=$PORT"
  echo "selected_port=$PORT"
  echo "selected_tm_port=$TM_PORT"
  echo "tm_port_offset=$TM_PORT_OFFSET"
  echo "stream_port_offset=$STREAM_PORT_OFFSET"
  echo "selected_service_ports=$SELECTED_SERVICE_PORTS"
  echo "port_selection_reason=$PORT_SELECTION_REASON"
  echo "port_selection_cleanup_closed=$PORT_SELECTION_CLEANUP_CLOSED"
  echo "port_selection_unmatched_count=$PORT_SELECTION_UNMATCHED_COUNT"
  echo "gpu=${GPU_VALUE:-<inherit>}"
  echo "mode_requested=$RUN_MODE"
  echo "diag_interval=$DIAG_INTERVAL"
  echo "startup_timeout_sec=$STARTUP_TIMEOUT_SEC"
  echo "core_wait_seconds=$CORE_WAIT_SECONDS"
  echo "stop_after_eval=$STOP_AFTER_EVAL"
  echo "repeat_eval_until_crash=$REPEAT_EVAL_UNTIL_CRASH"
  echo "repeat_eval_delay_seconds=$REPEAT_EVAL_DELAY_SECONDS"
  echo "max_eval_runs=$MAX_EVAL_RUNS"
  echo "eval_uses_run_custom_eval=$EVAL_USES_RUN_CUSTOM_EVAL"
  echo "eval_has_start_carla=$EVAL_HAS_START_CARLA"
  echo "eval_inferred_port=${EVAL_INFERRED_PORT:-}"
  echo "eval_inferred_tm_port=${EVAL_INFERRED_TM_PORT:-}"
  echo "eval_inferred_tm_port_offset=${EVAL_INFERRED_TM_PORT_OFFSET:-}"
  echo "eval_cmd_original=$EVAL_CMD"
  echo "eval_cmd_resolved=$EVAL_CMD_RESOLVED"
  echo "carla_extra_args=${CARLA_EXTRA_ARGS[*]}"
  echo "carla_runtime_log=$CARLA_RUNTIME_LOG"
  echo "carla_console_log=$CARLA_CONSOLE_LOG"
  echo "ulimit_open_files=$(ulimit -n)"
  echo "ulimit_core=$(ulimit -c)"
} >"$LOGDIR/run_meta.txt"

trap cleanup EXIT

capture_host_snapshot
capture_process_list "$LOGDIR/process_snapshot_start.txt"
capture_port_snapshot "$LOGDIR/port_snapshot_selected.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"$LOGDIR/nvidia_smi_start.txt" 2>&1 || true
fi

log_note "[ROOTCAUSE] logdir: $LOGDIR"
log_note "[ROOTCAUSE] using CARLA/TM ports $PORT/$TM_PORT (reason=$PORT_SELECTION_REASON)."

case "$RUN_MODE" in
  gdb)
    if ! launch_carla_under_gdb; then
      echo "[ERROR] $LAST_ERROR" >&2
      exit 1
    fi
    ;;
  core)
    if ! launch_carla_direct; then
      echo "[ERROR] $LAST_ERROR" >&2
      exit 1
    fi
    ;;
  auto)
    if ! launch_carla_under_gdb; then
      log_note "[WARN] gdb capture failed; falling back to direct core mode. reason=$LAST_ERROR"
      if pid_is_alive "${GDB_PID:-}"; then
        kill -INT "$GDB_PID" >/dev/null 2>&1 || true
        wait "$GDB_PID" >/dev/null 2>&1 || true
      fi
      GDB_PID=""
      CARLA_PID=""
      if ! launch_carla_direct; then
        echo "[ERROR] $LAST_ERROR" >&2
        exit 1
      fi
    fi
    ;;
esac

if ! start_diag_sidecar; then
  echo "[ERROR] $LAST_ERROR" >&2
  exit 1
fi

if [[ -n "$EVAL_CMD" ]]; then
  while :; do
    if capture_target_has_stopped; then
      EVAL_LOOP_END_REASON="target_stopped"
      log_note "[ROOTCAUSE] capture target stopped before eval run $((EVAL_RUN_COUNT + 1)); ending eval loop."
      break
    fi
    if (( MAX_EVAL_RUNS > 0 && EVAL_RUN_COUNT >= MAX_EVAL_RUNS )); then
      EVAL_LOOP_END_REASON="max_eval_runs"
      log_note "[ROOTCAUSE] reached max eval runs ($MAX_EVAL_RUNS); ending eval loop."
      break
    fi

    run_eval_once

    if (( STOP_AFTER_EVAL == 1 )); then
      EVAL_LOOP_END_REASON="stop_after_eval"
      stop_active_target
      break
    fi

    if capture_target_has_stopped; then
      EVAL_LOOP_END_REASON="target_stopped"
      log_note "[ROOTCAUSE] capture target stopped after eval run $EVAL_RUN_COUNT; ending eval loop."
      break
    fi

    if (( REPEAT_EVAL_UNTIL_CRASH == 0 )); then
      EVAL_LOOP_END_REASON="single_run"
      break
    fi

    if (( REPEAT_EVAL_DELAY_SECONDS > 0 )); then
      log_note "[ROOTCAUSE] target still alive after eval run $EVAL_RUN_COUNT; sleeping ${REPEAT_EVAL_DELAY_SECONDS}s before restart."
      sleep "$REPEAT_EVAL_DELAY_SECONDS"
    else
      log_note "[ROOTCAUSE] target still alive after eval run $EVAL_RUN_COUNT; restarting eval command immediately."
    fi
  done

  if [[ "$EVAL_LOOP_END_REASON" == "max_eval_runs" ]]; then
    stop_active_target
  fi
else
  log_note "[ROOTCAUSE] no --eval-cmd provided."
  log_note "[ROOTCAUSE] run your eval manually in another shell."
  log_note "[ROOTCAUSE] this script will keep collecting until CARLA exits."
fi

if [[ -n "$EVAL_LOOP_END_REASON" ]]; then
  echo "eval_loop_end_reason=$EVAL_LOOP_END_REASON" >>"$LOGDIR/run_meta.txt"
fi

if [[ "$TRACE_MODE_USED" == "gdb" ]]; then
  set +e
  wait "$GDB_PID"
  GDB_EXIT_CODE=$?
  set -e
  echo "gdb_exit_code_raw=$GDB_EXIT_CODE" >>"$LOGDIR/run_meta.txt"
  if (( INTENTIONAL_STOP == 1 && EVAL_EXIT_CODE == 0 )) \
    && [[ "$EVAL_LOOP_END_REASON" == "stop_after_eval" || "$EVAL_LOOP_END_REASON" == "max_eval_runs" ]]; then
    log_note "[ROOTCAUSE] normalizing gdb exit code after intentional wrapper shutdown."
    GDB_EXIT_CODE=0
  fi
  echo "gdb_exit_code=$GDB_EXIT_CODE" >>"$LOGDIR/run_meta.txt"
  if (( GDB_EXIT_CODE != 0 )); then
    log_note "[WARN] gdb exited with code $GDB_EXIT_CODE."
  fi
else
  set +e
  wait "$CARLA_PID"
  CARLA_EXIT_CODE=$?
  set -e
  echo "carla_exit_code_raw=$CARLA_EXIT_CODE" >>"$LOGDIR/run_meta.txt"
  if (( INTENTIONAL_STOP == 1 && EVAL_EXIT_CODE == 0 )) \
    && [[ "$EVAL_LOOP_END_REASON" == "stop_after_eval" || "$EVAL_LOOP_END_REASON" == "max_eval_runs" ]]; then
    log_note "[ROOTCAUSE] normalizing CARLA exit code after intentional wrapper shutdown."
    CARLA_EXIT_CODE=0
  fi
  echo "carla_exit_code=$CARLA_EXIT_CODE" >>"$LOGDIR/run_meta.txt"
  if (( CARLA_EXIT_CODE != 0 )); then
    log_note "[WARN] CARLA exited with code $CARLA_EXIT_CODE."
  fi
  CORE_FILE="$(wait_for_core_file || true)"
  if [[ -n "$CORE_FILE" ]]; then
    echo "core_file=$CORE_FILE" >>"$LOGDIR/run_meta.txt"
    run_postmortem_gdb "$CORE_FILE"
  else
    log_note "[WARN] No core file found under $CARLA_ROOT within ${CORE_WAIT_SECONDS}s after CARLA exited."
  fi
fi

log_note "[ROOTCAUSE] finished. Logs are in: $LOGDIR"
if (( GDB_EXIT_CODE != 0 )); then
  exit "$GDB_EXIT_CODE"
fi
if (( CARLA_EXIT_CODE != 0 )); then
  exit "$CARLA_EXIT_CODE"
fi
if (( EVAL_EXIT_CODE != 0 )); then
  exit "$EVAL_EXIT_CODE"
fi
exit 0
