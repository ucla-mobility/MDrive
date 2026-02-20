#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./download_and_setup_carla.sh [--validate] [--keep-base-archive]

Installs CARLA 0.9.12 into ./carla, imports additional maps + UCLA map,
applies targeted UCLA stability fixes, and optionally validates startup/map load.
EOF
}

log() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

die() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing dependency: $cmd"
}

is_valid_tar_gz() {
  local archive="$1"
  log "Validating archive integrity (this can take a few minutes): $archive"
  tar -tzf "$archive" >/dev/null 2>&1
}

download_tar_if_needed() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" ]]; then
    if is_valid_tar_gz "$out"; then
      log "Using existing archive: $out"
      return
    fi
    warn "Existing archive is invalid/corrupt, re-downloading: $out"
    rm -f "$out"
  fi

  local tmp="${out}.part"
  log "Downloading: $url"
  if ! wget -O "$tmp" "$url"; then
    rm -f "$tmp"
    die "Download failed: $url"
  fi
  mv "$tmp" "$out"

  is_valid_tar_gz "$out" || die "Downloaded archive is invalid: $out"
  log "Downloaded: $out"
}

copy_if_different() {
  local src="$1"
  local dst="$2"
  if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
    log "Unchanged file, keeping existing: $dst"
    return
  fi
  cp "$src" "$dst"
  log "Copied: $src -> $dst"
}

apply_ucla_package_fix() {
  local pkg="$1"
  python3 - "$pkg" <<'PY'
import json
import pathlib
import sys

pkg = pathlib.Path(sys.argv[1])
data = json.loads(pkg.read_text(encoding="utf-8"))
changed = False

for entry in data.get("maps", []):
    if entry.get("name") == "ucla_v2" and entry.get("use_carla_materials") is not True:
        entry["use_carla_materials"] = True
        changed = True

if changed:
    pkg.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    print("patched")
else:
    print("ok")
PY
}

apply_weather_binary_fixes() {
  local carla_dir="$1"
  python3 - "$carla_dir" <<'PY'
import hashlib
import sys
from pathlib import Path

carla_dir = Path(sys.argv[1])

# GUID bytes use Unreal serialization order (little-endian 32-bit words).
guid_4f37 = bytes.fromhex("bd3a374fdb4f3b08d2d06eba61095761")
guid_0aac = bytes.fromhex("aefdac0a7f24ab4cacffbb87f2e1d958")
content_root = carla_dir / "CarlaUE4/Content/Carla"
if not content_root.exists():
    raise FileNotFoundError(content_root)

patched_files = 0
patched_refs = 0
files = []
for pattern in ("*.uasset", "*.uexp"):
    files.extend(content_root.rglob(pattern))

total = len(files)
for idx, path in enumerate(files, start=1):
    if idx % 2000 == 0:
        print(f"guid_scan_progress {idx}/{total}")
        sys.stdout.flush()
    if "bak" in path.name:
        continue
    data = path.read_bytes()
    refs = data.count(guid_4f37)
    if refs == 0:
        continue
    path.write_bytes(data.replace(guid_4f37, guid_0aac))
    patched_files += 1
    patched_refs += refs

print(f"guid_scan_progress {total}/{total}")
print(f"patched_guid_files {patched_files}")
print(f"patched_guid_refs {patched_refs}")

# Align WeatherMaterialParameters.uasset with the known stable 0.9.12 patch bytes.
uasset = carla_dir / "CarlaUE4/Content/Carla/Blueprints/Weather/Materials/WeatherMaterialParameters.uasset"
if not uasset.exists():
    raise FileNotFoundError(uasset)

raw = bytearray(uasset.read_bytes())
if len(raw) < 109:
    raise RuntimeError(f"Unexpected WeatherMaterialParameters.uasset size: {len(raw)}")

# Header bytes + package GUID bytes observed in stable install.
raw[8] = 0x60
raw[9] = 0x03
raw[12] = 0x06
raw[13] = 0x02
raw[93:109] = bytes.fromhex("4254b59b69d87949b4c48b2cc9447153")
uasset.write_bytes(raw)

sha = hashlib.sha256(bytes(raw)).hexdigest()
print(f"weather_uasset_sha256 {sha}")

residual = 0
residual_checked = 0
for pattern in ("*.uasset", "*.uexp"):
    for path in content_root.rglob(pattern):
        residual_checked += 1
        if residual_checked % 3000 == 0:
            print(f"guid_verify_progress {residual_checked}")
            sys.stdout.flush()
        if "bak" in path.name:
            continue
        if guid_4f37 in path.read_bytes():
            residual += 1
if residual:
    raise RuntimeError(f"Residual active files still contain 4F37 GUID: {residual}")
PY
}

cache_core_weather_files() {
  local carla_archive="$1"
  local cache_uasset="$2"
  local cache_uexp="$3"

  if [[ -f "$cache_uasset" && -f "$cache_uexp" ]]; then
    log "Using cached core weather assets."
    return
  fi

  mkdir -p "$(dirname "$cache_uasset")"
  log "Caching core weather assets from CARLA base archive..."
  tar -xOf "$carla_archive" \
    "CarlaUE4/Content/Carla/Blueprints/Weather/Materials/WeatherMaterialParameters.uasset" \
    > "$cache_uasset"
  tar -xOf "$carla_archive" \
    "CarlaUE4/Content/Carla/Blueprints/Weather/Materials/WeatherMaterialParameters.uexp" \
    > "$cache_uexp"
  [[ -s "$cache_uasset" && -s "$cache_uexp" ]] || die "Failed to cache core weather assets."
  log "Cached core weather assets."
}

run_validation() {
  local carla_dir="$1"
  local repo_root="$2"
  local log_dir="$3"
  local port="${4:-2140}"

  local server_log="$log_dir/carla_validate_server.log"
  local client_log="$log_dir/carla_validate_client.log"

  log "Validation step 1/2: starting CARLA briefly (port $port)"
  "$carla_dir/CarlaUE4.sh" -RenderOffScreen -quality-level=Low -carla-rpc-port="$port" \
    >"$server_log" 2>&1 &
  local server_pid=$!

  cleanup_server() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }

  trap cleanup_server EXIT

  sleep 20
  if ! kill -0 "$server_pid" >/dev/null 2>&1; then
    die "CARLA exited early during validation. See: $server_log"
  fi
  log "CARLA stayed alive during startup window."

  local py37_bin=""
  if command -v python3.7 >/dev/null 2>&1; then
    py37_bin="python3.7"
  fi

  local egg="$carla_dir/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg"
  if [[ -n "$py37_bin" && -f "$egg" ]]; then
    log "Validation step 2/2: loading UCLA map via Python client"
    if ! "$py37_bin" "$repo_root/scripts/validate_carla_ucla_map.py" \
      --carla-root "$carla_dir" \
      --host 127.0.0.1 \
      --port "$port" \
      --map ucla_v2 >"$client_log" 2>&1; then
      die "Python validation failed. See: $client_log"
    fi
    log "Python client validation passed."
  else
    warn "Skipping Python client validation (requires python3.7 + CARLA py3.7 egg)."
  fi

  if grep -Eiq 'Segmentation fault|Fatal error|Unhandled Exception|Signal 11|Assertion failed|Failed to find parameter collection buffer with GUID' "$server_log"; then
    die "Fatal crash signature found in server log. See: $server_log"
  fi

  cleanup_server
  trap - EXIT
  log "Validation completed successfully. Logs: $log_dir"
}

main() {
  local do_validate="false"
  local keep_base_archive="false"
  if [[ $# -gt 2 ]]; then
    usage
    exit 1
  fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --validate) do_validate="true" ;;
      --keep-base-archive) keep_base_archive="true" ;;
      -h|--help) usage; exit 0 ;;
      *) usage; exit 1 ;;
    esac
    shift
  done

  require_cmd wget
  require_cmd tar
  require_cmd sha256sum
  require_cmd python3

  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$repo_root"

  local carla_dir="$repo_root/carla"
  local work_dir="$repo_root/.carla_work"
  local stamp_dir="$work_dir/stamps"
  local log_dir="$repo_root/logs"

  local carla_url="https://tiny.carla.org/carla-0-9-12-linux"
  local add_maps_url="https://tiny.carla.org/additional-maps-0-9-12-linux"

  local carla_archive="$work_dir/carla-0-9-12-linux"
  local cache_uasset="$work_dir/cache/WeatherMaterialParameters.uasset"
  local cache_uexp="$work_dir/cache/WeatherMaterialParameters.uexp"
  local ucla_src_archive="$repo_root/ucla_v2_0.9.12-dirty.tar.gz"
  local import_dir="$carla_dir/Import"
  local add_maps_archive="$import_dir/additional-maps-0-9-12-linux.tar.gz"
  local ucla_import_archive="$import_dir/ucla_v2_0.9.12-dirty.tar.gz"

  mkdir -p "$work_dir" "$stamp_dir" "$log_dir"
  [[ -f "$ucla_src_archive" ]] || die "Missing required UCLA archive: $ucla_src_archive"

  local has_carla_install="false"
  if [[ -f "$carla_dir/CarlaUE4.sh" ]]; then
    has_carla_install="true"
    log "CARLA already extracted at: $carla_dir"
  fi

  if [[ "$has_carla_install" == "false" || ! -f "$cache_uasset" || ! -f "$cache_uexp" ]]; then
    download_tar_if_needed "$carla_url" "$carla_archive"
  else
    log "Skipping CARLA base archive download; install and cache already present."
  fi

  if [[ "$has_carla_install" == "false" ]]; then
    if [[ -e "$carla_dir" ]]; then
      local backup_dir="$work_dir/incomplete_carla_$(date +%Y%m%d_%H%M%S)"
      warn "Found incomplete ./carla, moving aside: $backup_dir"
      mv "$carla_dir" "$backup_dir"
    fi
    mkdir -p "$carla_dir"
    log "Extracting CARLA into: $carla_dir"
    tar -xzf "$carla_archive" -C "$carla_dir"
    [[ -f "$carla_dir/CarlaUE4.sh" ]] || die "Extraction failed, CarlaUE4.sh not found"
  fi

  cache_core_weather_files "$carla_archive" "$cache_uasset" "$cache_uexp"

  mkdir -p "$import_dir"
  download_tar_if_needed "$add_maps_url" "$add_maps_archive"
  copy_if_different "$ucla_src_archive" "$ucla_import_archive"

  local import_signature
  log "Calculating import signature (this can take a few minutes)..."
  import_signature="$(sha256sum "$add_maps_archive" "$ucla_import_archive" | sha256sum | awk '{print $1}')"
  local import_stamp="$stamp_dir/import_assets.signature"
  local pkg_file="$carla_dir/CarlaUE4/Content/Carla/Config/ucla_v2.Package.json"
  local map_umap="$carla_dir/CarlaUE4/Content/Carla/Maps/ucla_v2/ucla_v2.umap"

  if [[ -f "$import_stamp" ]] && [[ "$(cat "$import_stamp")" == "$import_signature" ]] \
      && [[ -f "$pkg_file" ]] && [[ -f "$map_umap" ]]; then
    log "Import assets already up-to-date."
  else
    log "Importing additional maps + UCLA map (this may take a while)..."
    (
      cd "$carla_dir"
      ./ImportAssets.sh >"$log_dir/import_assets.log" 2>&1
    )
    [[ -f "$pkg_file" ]] || die "UCLA package file missing after import: $pkg_file"
    [[ -f "$map_umap" ]] || die "UCLA map asset missing after import: $map_umap"
    printf '%s\n' "$import_signature" >"$import_stamp"
    log "Import completed."
  fi

  # Fix A: WeatherParameters crash guard for UCLA map.
  # 1) Force use of CARLA base materials in UCLA package metadata.
  local pkg_fix_state
  pkg_fix_state="$(apply_ucla_package_fix "$pkg_file")"
  if [[ "$pkg_fix_state" == "patched" ]]; then
    log "Patched UCLA package metadata (use_carla_materials=true)."
  else
    log "UCLA package metadata already uses CARLA materials."
  fi

  # 2) Ensure the weather material parameter collection assets exist.
  local weather_uasset="$carla_dir/CarlaUE4/Content/Carla/Static/GenericMaterials/WetPavement/WeatherMaterialParameters.uasset"
  local weather_uexp="$carla_dir/CarlaUE4/Content/Carla/Static/GenericMaterials/WetPavement/WeatherMaterialParameters.uexp"
  if [[ ! -f "$weather_uasset" || ! -f "$weather_uexp" ]]; then
    log "Restoring WeatherMaterialParameters assets from UCLA archive."
    tar -xzf "$ucla_import_archive" -C "$carla_dir" \
      "CarlaUE4/Content/Carla/Static/GenericMaterials/WetPavement/WeatherMaterialParameters.uasset" \
      "CarlaUE4/Content/Carla/Static/GenericMaterials/WetPavement/WeatherMaterialParameters.uexp"
  fi
  [[ -f "$weather_uasset" && -f "$weather_uexp" ]] \
    || die "Missing WeatherMaterialParameters assets after restore."

  # 3) Restore CARLA core weather parameter collection used by weather blueprints.
  # UCLA archive can overwrite this with an incompatible GUID variant.
  local weather_bp_uasset="$carla_dir/CarlaUE4/Content/Carla/Blueprints/Weather/Materials/WeatherMaterialParameters.uasset"
  local weather_bp_uexp="$carla_dir/CarlaUE4/Content/Carla/Blueprints/Weather/Materials/WeatherMaterialParameters.uexp"
  copy_if_different "$cache_uasset" "$weather_bp_uasset"
  copy_if_different "$cache_uexp" "$weather_bp_uexp"
  [[ -f "$weather_bp_uasset" && -f "$weather_bp_uexp" ]] \
    || die "Missing blueprint WeatherMaterialParameters after core restore."
  log "Restored core Blueprint WeatherMaterialParameters assets."

  # 4) Apply deterministic binary GUID rewrites required by UCLA weather materials.
  log "Applying weather GUID fixes (this can take 1-3 minutes)..."
  apply_weather_binary_fixes "$carla_dir"
  log "Applied weather binary GUID fixes."

  # Fix B: XODR loading fix for UCLA map.
  local open_xodr="$carla_dir/CarlaUE4/Content/Carla/Maps/OpenDrive/ucla_v2.xodr"
  local map_xodr="$carla_dir/CarlaUE4/Content/Carla/Maps/ucla_v2/OpenDrive/ucla_v2.xodr"
  local open_xodr_backup="${open_xodr}.backup"
  [[ -f "$open_xodr" ]] || die "Missing source XODR after import: $open_xodr"
  mkdir -p "$(dirname "$map_xodr")"
  if [[ ! -f "$map_xodr" ]] || ! cmp -s "$open_xodr" "$map_xodr"; then
    cp "$open_xodr" "$map_xodr"
    log "Synced UCLA XODR into map-local OpenDrive path."
  else
    log "UCLA map-local XODR already in sync."
  fi
  if [[ ! -f "$open_xodr_backup" ]]; then
    cp "$open_xodr" "$open_xodr_backup"
    log "Created XODR backup: $open_xodr_backup"
  fi

  if [[ "$keep_base_archive" == "false" ]]; then
    if [[ -f "$carla_archive" ]]; then
      log "Removing downloaded CARLA base archive to save disk: $carla_archive"
      rm -f "$carla_archive"
    fi
  else
    log "Keeping CARLA base archive as requested: $carla_archive"
  fi

  log "Setup complete."
  log "CARLA root: $carla_dir"
  log "Logs dir: $log_dir"

  if [[ "$do_validate" == "true" ]]; then
    run_validation "$carla_dir" "$repo_root" "$log_dir"
  fi
}

main "$@"
