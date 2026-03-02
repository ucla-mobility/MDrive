#!/usr/bin/env python3
"""Audit recent debug pipeline runs and optionally run deferred CARLA validation.

This script can:
1) Scan debug run directories and summarize pass/fail status for the newest N runs.
2) Use already-running CARLA instances (host:port list) to run Stage-10 validation
   on eligible runs without launching/stopping CARLA itself.
3) Regenerate integrated pipeline dashboard (HTML + JSON) at the end.

Examples:
  python tools/audit_recent_debug_runs.py --root debug_runs --limit 500

  python tools/audit_recent_debug_runs.py --root debug_runs --limit 500 \
    --carla-instance 127.0.0.1:3000 --carla-instance 127.0.0.1:3006 \
    --json-out debug_runs/recent500_audit_after_carla.json
"""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import importlib.util
import json
import os
import random
import re
import signal
import socket
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

RUN_DIR_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{6})_.+")

DEFAULT_CARLA_REPAIR_MAX_ATTEMPTS = 2
DEFAULT_CARLA_REPAIR_XY_OFFSETS = "0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0"
DEFAULT_CARLA_REPAIR_Z_OFFSETS = "0.0,0.2,-0.2,0.5,-0.5,1.0"
DEFAULT_CARLA_TIMEOUT_S = 180.0

_AUTO_CARLA_MANAGERS: Set[Any] = set()
_AUTO_CARLA_MANAGERS_LOCK = threading.Lock()
_AUTO_CARLA_CLEANUP_INSTALLED = False
_CARLA_RPC_PROBE_MODULE: Optional[Any] = None
_CARLA_RPC_PROBE_LOAD_ERROR: Optional[str] = None

# Global state for resume checkpoint - allows saving on Ctrl+C
_RESUME_STATE_LOCK = threading.Lock()
_RESUME_STATE: Dict[str, Any] = {
    "path": None,           # Path to resume state file
    "done_runs": None,      # Set of completed run names
    "total_candidates": 0,  # Total candidates count
}


def _register_resume_state(path: Optional[Path], done_runs: Optional[Set[str]], total: int) -> None:
    """Register current resume state for signal handler to save on interrupt."""
    with _RESUME_STATE_LOCK:
        _RESUME_STATE["path"] = path
        _RESUME_STATE["done_runs"] = done_runs
        _RESUME_STATE["total_candidates"] = total


def _save_resume_state_on_interrupt() -> None:
    """Save current resume state - called by signal handler on Ctrl+C."""
    with _RESUME_STATE_LOCK:
        path = _RESUME_STATE.get("path")
        done_runs = _RESUME_STATE.get("done_runs")
        total = _RESUME_STATE.get("total_candidates", 0)
    
    if path is None or done_runs is None:
        return
    
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed_runs": sorted(str(x) for x in done_runs if str(x).strip()),
            "completed_count": len(done_runs),
            "total_candidates": int(total),
        }
        tmp_path = Path(str(path) + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        print(f"\n[INFO] Resume state saved on interrupt: {len(done_runs)} completed run(s) -> {path}")
    except Exception as exc:
        print(f"\n[WARN] Failed to save resume state on interrupt: {exc}")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_run_dir_name(text: str) -> Optional[str]:
    if not text:
        return None
    p = Path(text)
    parts = p.parts
    if "debug_runs" in parts:
        idx = parts.index("debug_runs")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if RUN_DIR_RE.match(candidate):
                return candidate
    candidate = p.name
    if RUN_DIR_RE.match(candidate):
        return candidate
    m = re.search(r"(\d{8}_\d{6}_\d{6}_[^/\\]+)", text)
    if m:
        return m.group(1)
    return None


def _run_sort_key(run_dir: Path) -> Tuple[str, str]:
    name = run_dir.name
    m = RUN_DIR_RE.match(name)
    if not m:
        return ("", name)
    return ("".join(m.groups()), name)


def _iter_run_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not RUN_DIR_RE.match(p.name):
            continue
        if (p / "summary.json").exists():
            out.append(p)
    out.sort(key=_run_sort_key, reverse=True)
    return out


def _load_dashboard_feature_map(root: Path) -> Dict[str, Dict[str, bool]]:
    """Map run_dir_name -> feature flags derived from *_dashboard*.json runs[] payloads."""
    feature_map: Dict[str, Dict[str, bool]] = {}
    for path in sorted(root.glob("*dashboard*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        runs = payload.get("runs")
        if not isinstance(runs, list):
            continue
        for row in runs:
            if not isinstance(row, dict):
                continue
            run_name = _extract_run_dir_name(str(row.get("run_dir", "")))
            if not run_name:
                continue
            flags = feature_map.setdefault(
                run_name,
                {
                    "similarity_scoring": False,
                    "interest_scoring": False,
                    "can_accept": False,
                    "duplicate_annotation": False,
                },
            )
            if any(k in row for k in ("similarity_best", "similarity_peer", "similarity_cluster_size")):
                flags["similarity_scoring"] = True
            if any(k in row for k in ("interest_score", "interest_score_adjusted")):
                flags["interest_scoring"] = True
            if "can_accept" in row:
                flags["can_accept"] = True
            if "duplicate_of" in row:
                flags["duplicate_annotation"] = True
    return feature_map


def _load_multi_run_result_map(root: Path) -> Dict[str, Dict[str, Any]]:
    """Map run_dir_name -> most recently seen result row from multi_run_*.json."""
    result_map: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.glob("multi_run_*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        rows = payload.get("results")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            run_name = _extract_run_dir_name(str(row.get("run_dir", "")))
            if run_name:
                result_map[run_name] = row
    return result_map


def _stage_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    stages = summary.get("stages")
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(stages, list):
        for row in stages:
            if isinstance(row, dict):
                name = str(row.get("stage", "")).strip()
                if name:
                    out[name] = row
    return out


def _is_carla_connect_failure_text(text: Optional[str]) -> bool:
    msg = str(text or "").strip().lower()
    if not msg:
        return False
    if "carla_connect_failed" in msg:
        return True
    if "carla connection" in msg and "failed" in msg:
        return True
    markers = (
        "resource temporarily unavailable",
        "connection refused",
        "connection reset",
        "network is unreachable",
        "timed out",
        "time-out",
    )
    return any(m in msg for m in markers)


def _is_carla_infra_pending(summary: Dict[str, Any], multi_row: Optional[Dict[str, Any]]) -> bool:
    if bool(summary.get("carla_validation_infra_error", False)):
        return True
    if _is_carla_connect_failure_text(summary.get("carla_validation_reason")):
        return True
    if _is_carla_connect_failure_text(summary.get("error_message")):
        return True

    smap = _stage_map(summary)
    carla_stage = smap.get("carla_validation")
    if isinstance(carla_stage, dict):
        warning = str(carla_stage.get("warning") or "").strip().lower()
        if warning == "carla_connect_failed_non_blocking":
            return True
        if _is_carla_connect_failure_text(carla_stage.get("infra_error")):
            return True
        if _is_carla_connect_failure_text(carla_stage.get("error")):
            return True

    if multi_row:
        if _is_carla_connect_failure_text(multi_row.get("error")):
            return True

    return False


def _infer_regular_pass(summary: Dict[str, Any], multi_row: Optional[Dict[str, Any]]) -> Optional[bool]:
    if isinstance(summary.get("validation_is_valid"), bool):
        return bool(summary.get("validation_is_valid"))

    failed_stage = summary.get("failed_stage")
    if isinstance(failed_stage, str) and failed_stage:
        return failed_stage != "validation"

    if multi_row and isinstance(multi_row.get("failed_stage"), str):
        failed_stage = str(multi_row.get("failed_stage"))
        if failed_stage:
            return failed_stage != "validation"
        if isinstance(multi_row.get("all_passed"), bool):
            return bool(multi_row.get("all_passed"))

    return None


def _infer_carla_pass(summary: Dict[str, Any], multi_row: Optional[Dict[str, Any]]) -> Optional[bool]:
    # CARLA connectivity outages are infra issues: keep scenario as pending CARLA.
    if _is_carla_infra_pending(summary, multi_row):
        return None

    if isinstance(summary.get("carla_validation_pass"), bool):
        return bool(summary.get("carla_validation_pass"))

    smap = _stage_map(summary)
    carla_stage = smap.get("carla_validation")
    if isinstance(carla_stage, dict) and isinstance(carla_stage.get("success"), bool):
        return bool(carla_stage.get("success"))

    if multi_row:
        failed_stage = multi_row.get("failed_stage")
        if isinstance(failed_stage, str) and failed_stage:
            if failed_stage == "carla_validation":
                return False
        err = str(multi_row.get("error") or "")
        if "carla_validation" in err.lower() or "carla validation" in err.lower():
            return False

    return None


def _failure_label(summary: Dict[str, Any], multi_row: Optional[Dict[str, Any]]) -> str:
    if summary.get("validation_is_valid") is False:
        return "validation:validation_is_valid_false"

    failed_stage = summary.get("failed_stage")
    if isinstance(failed_stage, str) and failed_stage:
        reason = str(summary.get("error_message") or "").strip()
        if reason:
            return f"{failed_stage}:{reason}"
        return failed_stage

    smap = _stage_map(summary)
    carla_stage = smap.get("carla_validation")
    if isinstance(carla_stage, dict) and carla_stage.get("success") is False:
        err = str(carla_stage.get("error") or "").strip()
        if err:
            return f"carla_validation:{err}"
        return "carla_validation"

    if multi_row:
        mstage = str(multi_row.get("failed_stage") or "").strip()
        merr = str(multi_row.get("error") or "").strip()
        if mstage:
            return f"{mstage}:{merr}" if merr else mstage

    return "unknown"


def _run_record(
    run_dir: Path,
    summary: Dict[str, Any],
    dashboard_flags: Dict[str, bool],
    multi_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    regular_pass = _infer_regular_pass(summary, multi_row)
    carla_pass = _infer_carla_pass(summary, multi_row)

    full_pass = bool(regular_pass is True and carla_pass is True)
    known_failure = bool((regular_pass is False) or (carla_pass is False))
    pending_carla = bool(regular_pass is True and carla_pass is None)

    target_meta = summary.get("target_acceptance")
    has_target_meta = isinstance(target_meta, dict)
    has_acceptance_level = has_target_meta and bool(target_meta.get("acceptance_level") is not None)
    has_geometry_fp = has_target_meta and bool(target_meta.get("geometry_fingerprint"))

    has_similarity = bool(dashboard_flags.get("similarity_scoring", False))
    has_interest = bool(dashboard_flags.get("interest_scoring", False))
    has_new_features = bool(
        has_target_meta
        or has_acceptance_level
        or has_geometry_fp
        or dashboard_flags.get("can_accept", False)
        or dashboard_flags.get("duplicate_annotation", False)
        or has_similarity
        or has_interest
    )

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "category": summary.get("category"),
        "seed": summary.get("seed"),
        "regular_pass": regular_pass,
        "carla_pass": carla_pass,
        "full_pass": full_pass,
        "known_failure": known_failure,
        "pending_carla": pending_carla,
        "failure_label": _failure_label(summary, multi_row) if known_failure else None,
        "validation_score": summary.get("validation_score"),
        "features": {
            "similarity_scoring": has_similarity,
            "interest_scoring": has_interest,
            "target_acceptance": has_target_meta,
            "acceptance_level": has_acceptance_level,
            "geometry_fingerprint": has_geometry_fp,
            "can_accept_flag": bool(dashboard_flags.get("can_accept", False)),
            "duplicate_annotation": bool(dashboard_flags.get("duplicate_annotation", False)),
            "any_new_feature": has_new_features,
        },
    }


def _pct(num: int, den: int) -> float:
    return (100.0 * num / den) if den else 0.0


def _summarize(records: List[Dict[str, Any]], *, ignore_carla_fails: bool = False) -> Dict[str, Any]:
    total = len(records)
    full_pass = sum(1 for r in records if r["full_pass"])
    known_failure = sum(1 for r in records if r["known_failure"])
    pending_carla = sum(1 for r in records if r["pending_carla"])

    regular_true = sum(1 for r in records if r["regular_pass"] is True)
    regular_false = sum(1 for r in records if r["regular_pass"] is False)
    regular_unknown = total - regular_true - regular_false

    carla_true = sum(1 for r in records if r["carla_pass"] is True)
    carla_false = sum(1 for r in records if r["carla_pass"] is False)
    carla_unknown = total - carla_true - carla_false

    feat_similarity = sum(1 for r in records if r["features"]["similarity_scoring"])
    feat_interest = sum(1 for r in records if r["features"]["interest_scoring"])
    feat_target = sum(1 for r in records if r["features"]["target_acceptance"])
    feat_new = sum(1 for r in records if r["features"]["any_new_feature"])

    failure_counts_strict: Counter[str] = Counter(
        str(r["failure_label"]) for r in records if r.get("failure_label")
    )

    if ignore_carla_fails:
        effective_full_pass_count = regular_true
        effective_known_failure_count = regular_false
        failure_counts_effective: Counter[str] = Counter()
        for r in records:
            if r.get("regular_pass") is False:
                label = str(r.get("failure_label") or "validation_failed")
                failure_counts_effective[label] += 1
        effective_policy = "regular_validation_only"
    else:
        effective_full_pass_count = full_pass
        effective_known_failure_count = known_failure
        failure_counts_effective = failure_counts_strict
        effective_policy = "regular_plus_carla"

    return {
        "total_runs": total,
        "effective_policy": effective_policy,
        "full_pass": {
            "count": full_pass,
            "rate_percent": round(_pct(full_pass, total), 2),
        },
        "known_failure": {
            "count": known_failure,
            "rate_percent": round(_pct(known_failure, total), 2),
        },
        "pending_carla": {
            "count": pending_carla,
            "rate_percent": round(_pct(pending_carla, total), 2),
        },
        "regular_validation": {
            "pass": regular_true,
            "fail": regular_false,
            "unknown": regular_unknown,
        },
        "carla_validation": {
            "pass": carla_true,
            "fail": carla_false,
            "unknown": carla_unknown,
        },
        "feature_coverage": {
            "similarity_scoring": feat_similarity,
            "interest_scoring": feat_interest,
            "target_acceptance": feat_target,
            "any_new_feature": feat_new,
        },
        "effective": {
            "full_pass": {
                "count": int(effective_full_pass_count),
                "rate_percent": round(_pct(int(effective_full_pass_count), total), 2),
            },
            "known_failure": {
                "count": int(effective_known_failure_count),
                "rate_percent": round(_pct(int(effective_known_failure_count), total), 2),
            },
        },
        "failure_breakdown": [
            {"label": k, "count": v} for k, v in failure_counts_strict.most_common()
        ],
        "effective_failure_breakdown": [
            {"label": k, "count": v} for k, v in failure_counts_effective.most_common()
        ],
    }


def _print_report(
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
    limit: int,
    root: Path,
    top_failures: int,
    ignore_carla_fails: bool,
) -> None:
    total_available = summary.get("total_available_runs", len(records))
    total = summary["total_runs"]
    print("=" * 72)
    print("RECENT DEBUG RUN AUDIT")
    print("=" * 72)
    print(f"Root: {root}")
    print(f"Requested newest runs: {limit}")
    print(f"Available runs with summary.json: {total_available}")
    print(f"Analyzed runs: {total}")

    if records:
        newest = records[0]["run_name"]
        oldest = records[-1]["run_name"]
        print(f"Newest analyzed run: {newest}")
        print(f"Oldest analyzed run: {oldest}")

    fp = summary["full_pass"]
    kf = summary["known_failure"]
    pc = summary["pending_carla"]
    eff = summary.get("effective", {})
    eff_fp = (eff.get("full_pass") or fp)
    eff_kf = (eff.get("known_failure") or kf)
    print()
    print(f"Full pass (strict regular + CARLA): {fp['count']}/{total} ({fp['rate_percent']:.2f}%)")
    print(f"Known failed (strict): {kf['count']}/{total} ({kf['rate_percent']:.2f}%)")
    print(f"Pending/unknown CARLA: {pc['count']}/{total} ({pc['rate_percent']:.2f}%)")
    if ignore_carla_fails:
        print(
            f"Effective full pass (ignore CARLA fails): {eff_fp['count']}/{total} "
            f"({eff_fp['rate_percent']:.2f}%)"
        )
        print(
            f"Effective failed (ignore CARLA fails): {eff_kf['count']}/{total} "
            f"({eff_kf['rate_percent']:.2f}%)"
        )

    rv = summary["regular_validation"]
    cv = summary["carla_validation"]
    print()
    print(f"Regular validation: pass={rv['pass']} fail={rv['fail']} unknown={rv['unknown']}")
    print(f"CARLA validation:   pass={cv['pass']} fail={cv['fail']} unknown={cv['unknown']}")

    fc = summary["feature_coverage"]
    print()
    print("Similarity/New-feature coverage:")
    print(f"  similarity scoring present: {fc['similarity_scoring']}/{total}")
    print(f"  interest scoring present:   {fc['interest_scoring']}/{total}")
    print(f"  target_acceptance present:  {fc['target_acceptance']}/{total}")
    print(f"  any new-feature signal:     {fc['any_new_feature']}/{total}")

    print()
    hdr = "Top effective failure reasons" if ignore_carla_fails else "Top failure reasons"
    print(f"{hdr} (up to {top_failures}):")
    source_key = "effective_failure_breakdown" if ignore_carla_fails else "failure_breakdown"
    rows = summary.get(source_key, [])[: max(0, top_failures)]
    if not rows:
        print("  (none)")
    else:
        for row in rows:
            print(f"  {row['count']:>4}  {row['label']}")


def _load_start_pipeline_module(repo_root: Path):
    module_path = repo_root / "scenario_generator" / "start_pipeline.py"
    spec = importlib.util.spec_from_file_location("_start_pipeline_for_audit", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load start_pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parse_carla_instance(raw: str) -> Tuple[str, int]:
    txt = str(raw).strip()
    if not txt:
        raise ValueError("Empty CARLA instance value")

    if ":" in txt:
        host_raw, port_raw = txt.rsplit(":", 1)
        host = host_raw.strip() or "127.0.0.1"
        port = int(port_raw.strip())
    else:
        host = "127.0.0.1"
        port = int(txt)

    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid CARLA port in '{raw}'")
    return host, port


def _cleanup_auto_carla_managers(reason: str = "shutdown") -> None:
    with _AUTO_CARLA_MANAGERS_LOCK:
        managers = list(_AUTO_CARLA_MANAGERS)
        _AUTO_CARLA_MANAGERS.clear()
    if not managers:
        return
    print(f"[INFO] Auto CARLA cleanup ({reason}): stopping {len(managers)} manager(s).")
    for mgr in managers:
        try:
            mgr.stop()
        except Exception as exc:
            print(f"[WARN] Auto CARLA cleanup stop failed: {exc}")


def _register_auto_carla_manager(mgr: Any) -> None:
    with _AUTO_CARLA_MANAGERS_LOCK:
        _AUTO_CARLA_MANAGERS.add(mgr)


def _unregister_auto_carla_manager(mgr: Any) -> None:
    with _AUTO_CARLA_MANAGERS_LOCK:
        _AUTO_CARLA_MANAGERS.discard(mgr)


def _install_auto_carla_cleanup_handlers() -> None:
    global _AUTO_CARLA_CLEANUP_INSTALLED
    if _AUTO_CARLA_CLEANUP_INSTALLED:
        return
    _AUTO_CARLA_CLEANUP_INSTALLED = True

    atexit.register(_cleanup_auto_carla_managers, "atexit")

    def _handler(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        # Save resume state before cleaning up
        _save_resume_state_on_interrupt()
        _cleanup_auto_carla_managers(f"signal:{signame}")
        raise SystemExit(128 + int(signum))

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


# Separate signal handler for resume state when not using auto-carla
_RESUME_SIGNAL_INSTALLED = False


def _install_resume_signal_handlers() -> None:
    """Install signal handlers to save resume state on Ctrl+C, even when not using auto-carla."""
    global _RESUME_SIGNAL_INSTALLED, _AUTO_CARLA_CLEANUP_INSTALLED
    if _RESUME_SIGNAL_INSTALLED or _AUTO_CARLA_CLEANUP_INSTALLED:
        # If auto-carla handlers are installed, they already handle resume state
        return
    _RESUME_SIGNAL_INSTALLED = True

    def _handler(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        _save_resume_state_on_interrupt()
        raise SystemExit(128 + int(signum))

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _is_port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except Exception:
        return False


def _load_carla_module_for_probe() -> Optional[Any]:
    global _CARLA_RPC_PROBE_MODULE, _CARLA_RPC_PROBE_LOAD_ERROR
    if _CARLA_RPC_PROBE_MODULE is not None:
        return _CARLA_RPC_PROBE_MODULE
    if _CARLA_RPC_PROBE_LOAD_ERROR is not None:
        return None
    try:
        from scenario_generator import carla_validation as cv  # type: ignore

        _CARLA_RPC_PROBE_MODULE = cv._load_carla_module()
        return _CARLA_RPC_PROBE_MODULE
    except Exception as exc:
        _CARLA_RPC_PROBE_LOAD_ERROR = str(exc)
        print(f"[WARN] CARLA RPC probe unavailable; falling back to port-only checks: {exc}")
        return None


def _is_carla_rpc_ready(
    host: str,
    port: int,
    *,
    timeout_s: float = 12.0,
    retries: int = 1,
    sleep_s: float = 1.0,
) -> Tuple[bool, str]:
    carla_mod = _load_carla_module_for_probe()
    if carla_mod is None:
        return True, "probe_unavailable"

    last_err = ""
    for _ in range(max(1, int(retries))):
        try:
            client = carla_mod.Client(str(host), int(port))
            client.set_timeout(float(timeout_s))
            world = client.get_world()
            _ = world.get_map().name
            return True, ""
        except Exception as exc:
            last_err = str(exc)
            if float(sleep_s) > 0:
                time.sleep(float(sleep_s))
    return False, last_err or "unknown_rpc_probe_error"


def _is_carla_port_set_available(
    host: str,
    world_port: int,
    *,
    stream_offset: int = 1,
    tm_offset: int = 5,
    timeout_s: float = 0.5,
) -> bool:
    world = int(world_port)
    stream = world + int(stream_offset)
    tm = world + int(tm_offset)
    if world <= 0 or tm > 65535 or stream > 65535:
        return False
    return (
        (not _is_port_open(host, world, timeout_s=timeout_s))
        and (not _is_port_open(host, stream, timeout_s=timeout_s))
        and (not _is_port_open(host, tm, timeout_s=timeout_s))
    )


def _pick_random_carla_world_port(
    *,
    port_min: int,
    port_max: int,
    tm_offset: int = 5,
) -> int:
    lo = int(port_min)
    hi = int(port_max) - int(tm_offset)
    if hi < lo:
        raise ValueError(
            f"Invalid CARLA auto port range: min={port_min}, max={port_max}, tm_offset={tm_offset}"
        )
    return int(random.randint(lo, hi))


def _find_random_available_carla_port(
    *,
    host: str,
    port_min: int,
    port_max: int,
    stream_offset: int,
    tm_offset: int,
    attempts: int,
    reserved_world_ports: Optional[Set[int]] = None,
) -> int:
    tries = max(1, int(attempts))
    reserved = reserved_world_ports or set()
    for _ in range(tries):
        world = _pick_random_carla_world_port(
            port_min=int(port_min),
            port_max=int(port_max),
            tm_offset=int(tm_offset),
        )
        if world in reserved:
            continue
        if _is_carla_port_set_available(
            host,
            world,
            stream_offset=int(stream_offset),
            tm_offset=int(tm_offset),
            timeout_s=0.5,
        ):
            return int(world)

    # Deterministic fallback scan through the whole range.
    lo = int(port_min)
    hi = int(port_max) - int(tm_offset)
    for world in range(lo, hi + 1):
        if world in reserved:
            continue
        if _is_carla_port_set_available(
            host,
            world,
            stream_offset=int(stream_offset),
            tm_offset=int(tm_offset),
            timeout_s=0.5,
        ):
            return int(world)

    raise RuntimeError(
        f"No available CARLA world port found in [{port_min}, {port_max}] on host {host}."
    )


def _select_carla_candidates(
    records: List[Dict[str, Any]],
    include_known_carla: bool,
    max_runs: Optional[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("regular_pass") is not True:
            continue
        if not include_known_carla and rec.get("carla_pass") is not None:
            continue
        out.append(rec)
    if max_runs is not None and max_runs >= 0:
        out = out[: max_runs]
    return out


def _record_run_name(rec: Dict[str, Any]) -> str:
    name = str(rec.get("run_name") or "").strip()
    if name:
        return name
    run_dir = str(rec.get("run_dir") or "").strip()
    if run_dir:
        return Path(run_dir).name
    return ""


def _load_resume_completed_runs(path: Path) -> Set[str]:
    payload = _read_json(path)
    if not payload:
        return set()
    rows = payload.get("completed_runs")
    if not isinstance(rows, list):
        return set()
    out: Set[str] = set()
    for row in rows:
        run_name = _extract_run_dir_name(str(row))
        if run_name:
            out.add(run_name)
    return out


def _has_definitive_carla_result(run_dir: Path) -> bool:
    """Check if a run has a definitive CARLA result (not an infra failure).
    
    Returns True only if the run has carla_validation_pass set AND is not an
    infrastructure failure (connection timeout, etc.). Runs that failed due to
    connectivity issues should be retried.
    """
    summary = _read_json(run_dir / "summary.json")
    if not summary:
        return False
    
    # Check if carla_validation_pass is set
    carla_pass = summary.get("carla_validation_pass")
    if carla_pass is None:
        return False  # No CARLA result at all
    
    # Check if this was an infrastructure failure (should be retryable)
    if _is_carla_infra_pending(summary, None):
        return False  # Infra failure - should retry
    
    return True  # Has definitive pass/fail result


def _write_resume_state(path: Path, completed_runs: Set[str], total_candidates: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_runs": sorted(str(x) for x in completed_runs if str(x).strip()),
        "completed_count": len(completed_runs),
        "total_candidates": int(total_candidates),
    }
    tmp_path = Path(str(path) + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _extract_carla_failure_reason(run_dir: Path, fallback: Optional[str] = None) -> str:
    summary = _read_json(run_dir / "summary.json") or {}
    reason = str(summary.get("carla_validation_reason") or "").strip()
    if reason:
        return reason
    err_msg = str(summary.get("error_message") or "").strip()
    if err_msg:
        return err_msg

    out = _read_json(run_dir / "10_carla_validation" / "output.json") or {}
    reason = str(out.get("failure_reason") or "").strip()
    if reason:
        return reason
    err = str(out.get("error") or "").strip()
    if err:
        return err

    if fallback:
        fb = str(fallback).strip()
        if fb:
            return fb
    return "unknown_failure_reason"


def _run_carla_validation_on_instances(
    *,
    sp: Any,
    candidates: List[Dict[str, Any]],
    instances: List[Tuple[str, int]],
    carla_repair_max_attempts: int,
    carla_repair_xy_offsets: str,
    carla_repair_z_offsets: str,
    carla_align_before_validate: bool,
    carla_require_risk: bool,
    carla_validation_timeout: float,
    resume_state_path: Optional[Path] = None,
    resume_completed_runs: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not candidates:
        return {}
    if not instances:
        raise ValueError("No CARLA instances provided")

    buckets: List[List[Dict[str, Any]]] = [[] for _ in instances]
    for idx, rec in enumerate(candidates):
        buckets[idx % len(instances)].append(rec)

    total = len(candidates)
    progress = {"done": 0}
    lock = threading.Lock()
    done_runs: Set[str] = set(str(x) for x in (resume_completed_runs or set()) if str(x).strip())
    resume_warned = {"write_failed": False}

    # Register resume state for signal handler to save on Ctrl+C
    if resume_state_path is not None:
        _register_resume_state(Path(resume_state_path), done_runs, total)
        _install_resume_signal_handlers()

    if resume_state_path is not None:
        try:
            _write_resume_state(Path(resume_state_path), done_runs, total)
        except Exception as exc:
            print(f"[WARN] Failed to initialize resume state at {resume_state_path}: {exc}")

    def _worker(instance_idx: int, host: str, port: int, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        updates: Dict[str, Dict[str, Any]] = {}
        for rec in items:
            run_dir = Path(str(rec["run_dir"]))
            run_name = run_dir.name
            category = rec.get("category")
            seed = rec.get("seed")
            try:
                updated = sp._run_deferred_carla_validation_for_run(
                    run_dir,
                    carla_host=str(host),
                    carla_port=int(port),
                    carla_repair_max_attempts=int(carla_repair_max_attempts),
                    carla_repair_xy_offsets=str(carla_repair_xy_offsets),
                    carla_repair_z_offsets=str(carla_repair_z_offsets),
                    carla_align_before_validate=bool(carla_align_before_validate),
                    carla_require_risk=bool(carla_require_risk),
                    carla_validation_timeout=float(carla_validation_timeout),
                    carla_process_started=False,
                )
                updated_rec = dict(updated) if isinstance(updated, dict) else {}
                connect_failure = bool(
                    updated_rec.get("failed_stage") == "carla_validation"
                    and _is_carla_connect_failure_text(updated_rec.get("error"))
                )

                if connect_failure and hasattr(sp, "_downgrade_connect_failure_to_infra_warning"):
                    try:
                        downgraded = sp._downgrade_connect_failure_to_infra_warning(
                            run_dir,
                            error=updated_rec.get("error"),
                            attempts=1,
                        )
                        if isinstance(downgraded, dict):
                            updated_rec = dict(downgraded)
                    except Exception:
                        # Keep original stage result if downgrade helper fails.
                        pass

                updates[run_name] = updated_rec
                if bool(updated_rec.get("carla_infra_warning")) or connect_failure:
                    status = "PENDING"
                    detail = "carla_connect_failed_non_blocking"
                elif bool(updated_rec.get("all_passed", False)):
                    status = "PASS"
                    detail = "carla_validation_pass"
                else:
                    status = "FAIL"
                    detail = _extract_carla_failure_reason(run_dir, fallback=str((updated or {}).get("error") or ""))
            except Exception as exc:
                updates[run_name] = {
                    "run_dir": str(run_dir),
                    "category": category,
                    "seed": seed,
                    "all_passed": False,
                    "failed_stage": "carla_validation",
                    "error": f"carla_validation_exception:{exc}",
                }
                status = "ERROR"
                detail = str(exc)

            with lock:
                if resume_state_path is not None and run_name:
                    done_runs.add(run_name)
                    try:
                        _write_resume_state(Path(resume_state_path), done_runs, total)
                    except Exception as exc:
                        if not resume_warned["write_failed"]:
                            print(f"[WARN] Failed writing resume state at {resume_state_path}: {exc}")
                            resume_warned["write_failed"] = True
                progress["done"] += 1
                done = int(progress["done"])
                detail_short = (detail[:240] + "...") if len(detail) > 240 else detail
                print(
                    f"  [CARLA {done}/{total}] {status} {category} (seed={seed}) "
                    f"via {host}:{port} run={run_name} reason={detail_short}".strip()
                )
        return updates

    merged: Dict[str, Dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(instances)) as pool:
        future_meta: Dict[concurrent.futures.Future, str] = {}
        for idx, (inst, items) in enumerate(zip(instances, buckets), start=1):
            host, port = inst
            if not items:
                continue
            print(f"[INFO] CARLA worker {idx}: {host}:{port} assigned {len(items)} run(s)")
            fut = pool.submit(_worker, idx, host, port, items)
            future_meta[fut] = f"{host}:{port}"

        for fut in concurrent.futures.as_completed(list(future_meta.keys())):
            worker_name = future_meta.get(fut, "unknown")
            try:
                merged.update(fut.result())
            except Exception as exc:
                print(f"[WARN] CARLA worker {worker_name} crashed; continuing with completed results: {exc}")

    return merged


def _run_carla_validation_with_auto_instances(
    *,
    sp: Any,
    candidates: List[Dict[str, Any]],
    worker_count: int,
    carla_host: str,
    carla_root: Optional[Path],
    carla_args: List[str],
    carla_auto_port_min: int,
    carla_auto_port_max: int,
    carla_connect_retries: int,
    carla_start_wait_s: float,
    carla_rpc_probe_timeout_s: float,
    carla_rpc_probe_retries: int,
    carla_repair_max_attempts: int,
    carla_repair_xy_offsets: str,
    carla_repair_z_offsets: str,
    carla_align_before_validate: bool,
    carla_require_risk: bool,
    carla_validation_timeout: float,
    resume_state_path: Optional[Path] = None,
    resume_completed_runs: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    if not candidates:
        return {}

    manager_cls = getattr(sp, "CarlaProcessManager", None)
    if manager_cls is None:
        raise RuntimeError("start_pipeline.CarlaProcessManager is unavailable.")
    _install_auto_carla_cleanup_handlers()

    repo_root = Path(__file__).resolve().parents[1]
    resolved_carla_root = (
        Path(carla_root).expanduser().resolve()
        if carla_root is not None
        else Path(os.environ.get("CARLA_ROOT", str(repo_root / "carla912"))).expanduser().resolve()
    )

    resolved_carla_args = [str(a).strip() for a in (carla_args or []) if str(a).strip()]
    lowered_renderer = {a.lower() for a in resolved_carla_args}
    if not ({"-opengl", "-vulkan", "-nullrhi"} & lowered_renderer):
        resolved_carla_args.append("-opengl")
    if hasattr(sp, "_effective_carla_args"):
        try:
            resolved_carla_args = list(sp._effective_carla_args(resolved_carla_args))
        except Exception:
            pass

    startup_timeout_s = float(getattr(sp, "CARLA_STARTUP_TIMEOUT_S", 120.0))
    shutdown_timeout_s = float(getattr(sp, "CARLA_SHUTDOWN_TIMEOUT_S", 15.0))
    port_tries = int(max(8, int(getattr(sp, "CARLA_PORT_TRIES", 8))))
    tm_offset = int(getattr(sp, "CARLA_TM_PORT_OFFSET", 5))
    stream_offset = int(getattr(sp, "CARLA_STREAM_PORT_OFFSET", 1))
    port_step = int(max(tm_offset + 1, int(getattr(sp, "CARLA_PORT_STEP", 1))))
    connect_retries = max(0, int(carla_connect_retries))

    workers = max(1, int(worker_count))
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(workers)]
    for idx, rec in enumerate(candidates):
        buckets[idx % workers].append(rec)

    total = len(candidates)
    progress = {"done": 0}
    done_runs: Set[str] = set(str(x) for x in (resume_completed_runs or set()) if str(x).strip())
    resume_warned = {"write_failed": False}
    progress_lock = threading.Lock()
    port_lock = threading.Lock()
    reserved_world_ports: Set[int] = set()

    # Register resume state for signal handler to save on Ctrl+C
    if resume_state_path is not None:
        _register_resume_state(Path(resume_state_path), done_runs, total)

    if resume_state_path is not None:
        try:
            _write_resume_state(Path(resume_state_path), done_runs, total)
        except Exception as exc:
            print(f"[WARN] Failed to initialize resume state at {resume_state_path}: {exc}")

    def _worker(worker_idx: int, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        updates: Dict[str, Dict[str, Any]] = {}
        manager: Optional[Any] = None
        active_port: Optional[int] = None

        def _release_active_port() -> None:
            nonlocal active_port
            if active_port is None:
                return
            with port_lock:
                reserved_world_ports.discard(int(active_port))
            active_port = None

        def _start_or_restart(reason: str) -> int:
            nonlocal manager, active_port
            if manager is not None:
                try:
                    manager.stop()
                except Exception:
                    pass
                _unregister_auto_carla_manager(manager)
                manager = None
            _release_active_port()

            with port_lock:
                requested_world_port = _find_random_available_carla_port(
                    host=str(carla_host),
                    port_min=int(carla_auto_port_min),
                    port_max=int(carla_auto_port_max),
                    stream_offset=int(stream_offset),
                    tm_offset=int(tm_offset),
                    attempts=max(32, int(port_tries) * 8),
                    reserved_world_ports=set(reserved_world_ports),
                )
                reserved_world_ports.add(int(requested_world_port))

            mgr = manager_cls(
                carla_root=resolved_carla_root,
                host=str(carla_host),
                port=int(requested_world_port),
                extra_args=list(resolved_carla_args),
                startup_timeout_s=float(startup_timeout_s),
                shutdown_timeout_s=float(shutdown_timeout_s),
                port_tries=int(port_tries),
                port_step=int(port_step),
            )
            try:
                selected_world_port = int(mgr.start())
            except Exception:
                with port_lock:
                    reserved_world_ports.discard(int(requested_world_port))
                raise

            with port_lock:
                reserved_world_ports.discard(int(requested_world_port))
                if selected_world_port in reserved_world_ports:
                    try:
                        mgr.stop()
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Auto-selected CARLA world port collision detected: {selected_world_port}"
                    )
                reserved_world_ports.add(int(selected_world_port))

            manager = mgr
            _register_auto_carla_manager(manager)
            active_port = int(selected_world_port)
            print(
                f"[INFO] CARLA auto worker {worker_idx}: started {carla_host}:{active_port} "
                f"({reason})"
            )
            if float(carla_start_wait_s) > 0:
                time.sleep(float(carla_start_wait_s))
            rpc_ok, rpc_err = _is_carla_rpc_ready(
                str(carla_host),
                int(active_port),
                timeout_s=float(carla_rpc_probe_timeout_s),
                retries=max(1, int(carla_rpc_probe_retries)),
                sleep_s=1.0,
            )
            if not rpc_ok:
                try:
                    manager.stop()
                except Exception:
                    pass
                _unregister_auto_carla_manager(manager)
                manager = None
                _release_active_port()
                raise RuntimeError(f"rpc_probe_failed:{rpc_err}")
            return int(active_port)

        try:
            if items:
                try:
                    _start_or_restart("initial")
                except Exception as exc:
                    err_text = f"carla_connect_failed:{exc}"
                    print(
                        f"[WARN] CARLA auto worker {worker_idx}: initial launch failed; "
                        f"marking {len(items)} assigned run(s) as pending CARLA infra error: {exc}"
                    )
                    for rec in items:
                        run_dir = Path(str(rec["run_dir"]))
                        run_name = run_dir.name
                        category = rec.get("category")
                        seed = rec.get("seed")
                        updated_rec: Dict[str, Any] = {
                            "run_dir": str(run_dir),
                            "category": category,
                            "seed": seed,
                            "all_passed": False,
                            "failed_stage": "carla_validation",
                            "error": err_text,
                            "carla_infra_warning": True,
                        }
                        if hasattr(sp, "_downgrade_connect_failure_to_infra_warning"):
                            try:
                                downgraded = sp._downgrade_connect_failure_to_infra_warning(
                                    run_dir,
                                    error=err_text,
                                    attempts=1,
                                )
                                if isinstance(downgraded, dict):
                                    updated_rec = dict(downgraded)
                            except Exception:
                                pass
                        updates[run_name] = updated_rec
                        with progress_lock:
                            if resume_state_path is not None and run_name:
                                done_runs.add(run_name)
                                try:
                                    _write_resume_state(Path(resume_state_path), done_runs, total)
                                except Exception as write_exc:
                                    if not resume_warned["write_failed"]:
                                        print(f"[WARN] Failed writing resume state at {resume_state_path}: {write_exc}")
                                        resume_warned["write_failed"] = True
                            progress["done"] += 1
                            done = int(progress["done"])
                            print(
                                f"  [CARLA {done}/{total}] PENDING {category} (seed={seed}) "
                                f"via {carla_host}:unavailable run={run_name} "
                                "reason=carla_connect_failed_non_blocking".strip()
                            )
                    return updates
            for rec in items:
                run_dir = Path(str(rec["run_dir"]))
                run_name = run_dir.name
                category = rec.get("category")
                seed = rec.get("seed")

                attempts_used = 0
                final_connect_failure = False
                updated_rec: Dict[str, Any] = {
                    "run_dir": str(run_dir),
                    "category": category,
                    "seed": seed,
                    "all_passed": False,
                    "failed_stage": "carla_validation",
                    "error": "carla_connect_failed:not_started",
                }
                for infra_try in range(connect_retries + 1):
                    attempts_used = infra_try + 1
                    print(
                        f"[INFO] CARLA auto worker {worker_idx}: "
                        f"run={run_name} attempt={attempts_used}/{connect_retries + 1} "
                        f"port={active_port if active_port is not None else 'none'}"
                    )
                    needs_restart = (
                        manager is None
                        or active_port is None
                        or (hasattr(manager, "is_running") and not bool(manager.is_running()))
                        or not _is_port_open(str(carla_host), int(active_port), timeout_s=1.0)
                    )
                    if not needs_restart and active_port is not None:
                        rpc_ok, rpc_err = _is_carla_rpc_ready(
                            str(carla_host),
                            int(active_port),
                            timeout_s=max(5.0, float(carla_rpc_probe_timeout_s) * 0.75),
                            retries=1,
                            sleep_s=0.0,
                        )
                        if not rpc_ok:
                            print(
                                f"[WARN] CARLA auto worker {worker_idx}: "
                                f"rpc_probe_failed_before_validation: {rpc_err}"
                            )
                            needs_restart = True
                    if needs_restart:
                        reason = (
                            "healthcheck_failed_before_validation"
                            if infra_try == 0
                            else "retry_healthcheck_failed_before_validation"
                        )
                        print(f"[WARN] CARLA auto worker {worker_idx}: {reason}; relaunching.")
                        try:
                            _start_or_restart(reason)
                        except Exception as exc:
                            updated_rec = {
                                "run_dir": str(run_dir),
                                "category": category,
                                "seed": seed,
                                "all_passed": False,
                                "failed_stage": "carla_validation",
                                "error": f"carla_connect_failed:{exc}",
                            }
                            final_connect_failure = True
                            if infra_try < connect_retries:
                                continue
                            break

                    try:
                        updated = sp._run_deferred_carla_validation_for_run(
                            run_dir,
                            carla_host=str(carla_host),
                            carla_port=int(active_port),
                            carla_repair_max_attempts=int(carla_repair_max_attempts),
                            carla_repair_xy_offsets=str(carla_repair_xy_offsets),
                            carla_repair_z_offsets=str(carla_repair_z_offsets),
                            carla_align_before_validate=bool(carla_align_before_validate),
                            carla_require_risk=bool(carla_require_risk),
                            carla_validation_timeout=float(carla_validation_timeout),
                            carla_process_started=True,
                        )
                        updated_rec = dict(updated) if isinstance(updated, dict) else {}
                    except Exception as exc:
                        updated_rec = {
                            "run_dir": str(run_dir),
                            "category": category,
                            "seed": seed,
                            "all_passed": False,
                            "failed_stage": "carla_validation",
                            "error": f"carla_validation_exception:{exc}",
                        }

                    connect_failure = bool(
                        updated_rec.get("failed_stage") == "carla_validation"
                        and _is_carla_connect_failure_text(updated_rec.get("error"))
                    )
                    final_connect_failure = bool(connect_failure)
                    if connect_failure and infra_try < connect_retries:
                        print(
                            f"[WARN] CARLA auto worker {worker_idx}: post_validation_connect_failure; "
                            f"relaunching and retrying same run (attempt {attempts_used}/{connect_retries + 1})."
                        )
                        try:
                            _start_or_restart("post_validation_connect_failure")
                        except Exception as exc:
                            updated_rec = {
                                "run_dir": str(run_dir),
                                "category": category,
                                "seed": seed,
                                "all_passed": False,
                                "failed_stage": "carla_validation",
                                "error": f"carla_connect_failed:{exc}",
                            }
                            final_connect_failure = True
                        continue
                    break

                if final_connect_failure and hasattr(sp, "_downgrade_connect_failure_to_infra_warning"):
                    try:
                        downgraded = sp._downgrade_connect_failure_to_infra_warning(
                            run_dir,
                            error=updated_rec.get("error"),
                            attempts=attempts_used,
                        )
                        if isinstance(downgraded, dict):
                            updated_rec = dict(downgraded)
                    except Exception:
                        pass

                updates[run_name] = updated_rec
                if bool(updated_rec.get("carla_infra_warning")) or final_connect_failure:
                    status = "PENDING"
                    detail = "carla_connect_failed_non_blocking"
                elif bool(updated_rec.get("all_passed", False)):
                    status = "PASS"
                    detail = "carla_validation_pass"
                else:
                    status = "FAIL"
                    detail = _extract_carla_failure_reason(
                        run_dir, fallback=str(updated_rec.get("error") or "")
                    )

                with progress_lock:
                    if resume_state_path is not None and run_name:
                        done_runs.add(run_name)
                        try:
                            _write_resume_state(Path(resume_state_path), done_runs, total)
                        except Exception as exc:
                            if not resume_warned["write_failed"]:
                                print(f"[WARN] Failed writing resume state at {resume_state_path}: {exc}")
                                resume_warned["write_failed"] = True
                    progress["done"] += 1
                    done = int(progress["done"])
                    detail_short = (detail[:240] + "...") if len(detail) > 240 else detail
                    print(
                        f"  [CARLA {done}/{total}] {status} {category} (seed={seed}) "
                        f"via {carla_host}:{active_port} run={run_name} reason={detail_short}".strip()
                    )
        finally:
            if manager is not None:
                try:
                    manager.stop()
                except Exception:
                    pass
                _unregister_auto_carla_manager(manager)
            _release_active_port()

        return updates

    merged: Dict[str, Dict[str, Any]] = {}
    active_workers = sum(1 for x in buckets if x)
    if active_workers <= 0:
        return merged
    with concurrent.futures.ThreadPoolExecutor(max_workers=active_workers) as pool:
        future_meta: Dict[concurrent.futures.Future, int] = {}
        for idx, items in enumerate(buckets, start=1):
            if not items:
                continue
            print(f"[INFO] CARLA auto worker {idx}: assigned {len(items)} run(s)")
            fut = pool.submit(_worker, idx, items)
            future_meta[fut] = idx

        for fut in concurrent.futures.as_completed(list(future_meta.keys())):
            worker_idx = future_meta.get(fut, -1)
            try:
                merged.update(fut.result())
            except Exception as exc:
                print(
                    f"[WARN] CARLA auto worker {worker_idx} crashed; "
                    f"continuing with completed results: {exc}"
                )
    return merged


def _build_records(
    selected_run_dirs: List[Path],
    dashboard_map: Dict[str, Dict[str, bool]],
    multi_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for run_dir in selected_run_dirs:
        summary = _read_json(run_dir / "summary.json")
        if not summary:
            continue
        flags = dashboard_map.get(run_dir.name, {})
        multi_row = multi_map.get(run_dir.name)
        records.append(_run_record(run_dir, summary, flags, multi_row))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit most recent debug runs; optionally run CARLA validation on external or auto-started instances."
    )
    parser.add_argument("--root", type=Path, default=Path("debug_runs"), help="Debug run root directory.")
    parser.add_argument("--limit", type=int, default=500, help="Number of most recent runs to analyze.")
    parser.add_argument("--top-failures", type=int, default=12, help="How many failure reasons to print.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output path for JSON report.")
    parser.add_argument(
        "--ignore-carla-fails",
        action="store_true",
        help="Treat CARLA failures as non-blocking for effective pass/fail counts (regular validation only).",
    )

    parser.add_argument(
        "--carla-instance",
        action="append",
        default=[],
        help="Existing CARLA endpoint host:port. Repeat for multiple instances.",
    )
    parser.add_argument(
        "--run-carla-validation",
        action="store_true",
        help="Run deferred CARLA validation using provided --carla-instance endpoints.",
    )
    parser.add_argument(
        "--carla-include-known",
        action="store_true",
        help="Re-run CARLA validation even for runs that already have CARLA pass/fail status.",
    )
    parser.add_argument(
        "--carla-max-runs",
        type=int,
        default=None,
        help="Optional cap on number of selected runs to send to CARLA validation.",
    )
    parser.add_argument("--carla-repair-max-attempts", type=int, default=DEFAULT_CARLA_REPAIR_MAX_ATTEMPTS)
    parser.add_argument("--carla-repair-xy-offsets", type=str, default=DEFAULT_CARLA_REPAIR_XY_OFFSETS)
    parser.add_argument("--carla-repair-z-offsets", type=str, default=DEFAULT_CARLA_REPAIR_Z_OFFSETS)
    parser.add_argument("--carla-align-before-validate", action="store_true")
    parser.add_argument("--carla-require-risk", dest="carla_require_risk", action="store_true")
    parser.add_argument("--no-carla-require-risk", dest="carla_require_risk", action="store_false")
    parser.set_defaults(carla_require_risk=True)
    parser.add_argument("--carla-validation-timeout", type=float, default=DEFAULT_CARLA_TIMEOUT_S)
    parser.add_argument(
        "--carla-resume",
        action="store_true",
        help="Resume CARLA reruns after interruption by checkpointing completed run names.",
    )
    parser.add_argument(
        "--carla-resume-state",
        type=Path,
        default=None,
        help="Checkpoint file path used by --carla-resume (default: <root>/.carla_validation_resume_state.json).",
    )
    parser.add_argument(
        "--carla-resume-reset",
        action="store_true",
        help="Reset existing CARLA resume checkpoint before processing.",
    )
    parser.add_argument(
        "--carla-auto-start",
        action="store_true",
        help="Auto-launch/manage local CARLA workers and restart them on connectivity drops.",
    )
    parser.add_argument(
        "--carla-auto-workers",
        type=int,
        default=2,
        help="Number of auto-managed CARLA worker processes to run in parallel.",
    )
    parser.add_argument(
        "--carla-auto-host",
        type=str,
        default="127.0.0.1",
        help="Host used for auto-managed CARLA instances.",
    )
    parser.add_argument(
        "--carla-auto-port-min",
        type=int,
        default=3000,
        help="Minimum world port for auto-managed CARLA instances.",
    )
    parser.add_argument(
        "--carla-auto-port-max",
        type=int,
        default=3999,
        help="Maximum world port for auto-managed CARLA instances.",
    )
    parser.add_argument(
        "--carla-auto-connect-retries",
        type=int,
        default=3,
        help="Per-run retries after CARLA connectivity failures in auto-start mode.",
    )
    parser.add_argument(
        "--carla-auto-start-wait",
        type=float,
        default=1.5,
        help="Short wait in seconds after each CARLA auto-launch before validation continues.",
    )
    parser.add_argument(
        "--carla-auto-rpc-probe-timeout",
        type=float,
        default=12.0,
        help="Timeout (seconds) for CARLA RPC readiness probe (client.get_world) in auto mode.",
    )
    parser.add_argument(
        "--carla-auto-rpc-probe-retries",
        type=int,
        default=2,
        help="How many RPC probe attempts to make after each auto-launch before considering it unhealthy.",
    )
    parser.add_argument(
        "--carla-root",
        type=Path,
        default=None,
        help="CARLA install root containing CarlaUE4.sh (default: $CARLA_ROOT or <repo>/carla912).",
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Extra argument passed to auto-launched CARLA instances; repeatable.",
    )

    parser.add_argument(
        "--build-dashboard",
        dest="build_dashboard",
        action="store_true",
        help="Build integrated pipeline dashboard after optional CARLA validation.",
    )
    parser.add_argument("--no-build-dashboard", dest="build_dashboard", action="store_false")
    parser.set_defaults(build_dashboard=True)
    parser.add_argument("--dashboard-html", type=str, default=None, help="Output HTML path for dashboard.")
    parser.add_argument("--dashboard-json", type=str, default=None, help="Output JSON path for dashboard payload.")
    parser.add_argument("--dashboard-title", type=str, default=None, help="Dashboard title override.")
    parser.add_argument(
        "--dashboard-glob",
        action="append",
        default=None,
        help="Optional extra glob(s) to include in dashboard, repeatable.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Debug root does not exist: {root}")

    run_dirs = _iter_run_dirs(root)
    selected_run_dirs = run_dirs[: max(0, int(args.limit))]

    dashboard_map = _load_dashboard_feature_map(root)
    multi_map = _load_multi_run_result_map(root)
    records = _build_records(selected_run_dirs, dashboard_map, multi_map)

    run_carla = bool(args.run_carla_validation or args.carla_instance or args.carla_auto_start)
    auto_carla_mode = bool(args.carla_auto_start)
    dashboard_info = None
    carla_updates: Dict[str, Dict[str, Any]] = {}
    resume_enabled = bool(args.carla_resume or args.carla_resume_state)
    resume_state_path: Optional[Path] = None
    resume_completed_runs: Set[str] = set()
    parsed_instances: List[Tuple[str, int]] = []

    if run_carla:
        if auto_carla_mode and args.carla_instance:
            raise SystemExit(
                "Use either --carla-auto-start or --carla-instance, not both at the same time."
            )

        if auto_carla_mode:
            if int(args.carla_auto_workers) <= 0:
                raise SystemExit("--carla-auto-workers must be >= 1.")
            if int(args.carla_auto_port_min) <= 0 or int(args.carla_auto_port_max) <= 0:
                raise SystemExit("--carla-auto-port-min/max must be positive integers.")
            if int(args.carla_auto_port_max) <= int(args.carla_auto_port_min):
                raise SystemExit("--carla-auto-port-max must be greater than --carla-auto-port-min.")
            print("\n" + "#" * 72)
            print("AUTO-START CARLA VALIDATION")
            print("#" * 72)
            print(
                f"  workers={int(args.carla_auto_workers)} "
                f"host={args.carla_auto_host} "
                f"port_range=[{int(args.carla_auto_port_min)},{int(args.carla_auto_port_max)}] "
                f"connect_retries={int(args.carla_auto_connect_retries)} "
                f"launch_wait_s={float(args.carla_auto_start_wait):.2f} "
                f"rpc_probe={float(args.carla_auto_rpc_probe_timeout):.1f}s x{max(1, int(args.carla_auto_rpc_probe_retries))}"
            )
        else:
            seen_instances = set()
            for raw in args.carla_instance:
                host, port = _parse_carla_instance(raw)
                key = (host, int(port))
                if key in seen_instances:
                    continue
                seen_instances.add(key)
                parsed_instances.append(key)

            if not parsed_instances:
                raise SystemExit("CARLA validation requested but no --carla-instance endpoints were provided.")

            print("\n" + "#" * 72)
            print("EXTERNAL CARLA VALIDATION")
            print("#" * 72)
            for host, port in parsed_instances:
                reachable = _is_port_open(host, port, timeout_s=1.0)
                status = "reachable" if reachable else "not reachable (connection test failed)"
                print(f"  {host}:{port} -> {status}")

        candidates = _select_carla_candidates(
            records,
            include_known_carla=bool(args.carla_include_known),
            max_runs=args.carla_max_runs,
        )

        if resume_enabled:
            resume_state_path = (
                Path(args.carla_resume_state).resolve()
                if args.carla_resume_state is not None
                else (root / ".carla_validation_resume_state.json")
            )
            if bool(args.carla_resume_reset) and resume_state_path.exists():
                try:
                    resume_state_path.unlink()
                    print(f"[INFO] CARLA resume checkpoint reset: {resume_state_path}")
                except Exception as exc:
                    print(f"[WARN] Failed to reset resume checkpoint {resume_state_path}: {exc}")
            if resume_state_path.exists():
                resume_completed_runs = _load_resume_completed_runs(resume_state_path)
            if resume_completed_runs:
                before = len(candidates)
                # Filter: skip only runs that are in resume AND have definitive results.
                # Runs with infra failures (connection timeouts, etc.) should be retried.
                filtered_candidates = []
                infra_retry_count = 0
                skipped_definitive = 0
                for rec in candidates:
                    run_name = _record_run_name(rec)
                    if run_name not in resume_completed_runs:
                        # Not in resume state - process it
                        filtered_candidates.append(rec)
                    else:
                        # In resume state - check if it has definitive result
                        run_dir = Path(rec.get("run_dir", ""))
                        if run_dir.exists() and _has_definitive_carla_result(run_dir):
                            # Definitive result - skip
                            skipped_definitive += 1
                        else:
                            # Infra failure or no result - retry
                            filtered_candidates.append(rec)
                            infra_retry_count += 1
                candidates = filtered_candidates
                print(
                    f"[INFO] CARLA resume enabled: skipping {skipped_definitive} run(s) with definitive results, "
                    f"retrying {infra_retry_count} infra-failure run(s) "
                    f"from {resume_state_path}"
                )
            else:
                print(f"[INFO] CARLA resume enabled: checkpoint={resume_state_path}")

        print(
            f"[INFO] CARLA candidate runs: {len(candidates)} "
            f"(from selected={len(records)}, include_known={bool(args.carla_include_known)})"
        )

        sp = None
        if candidates or bool(args.build_dashboard):
            repo_root = Path(__file__).resolve().parents[1]
            sp = _load_start_pipeline_module(repo_root)

        if candidates:
            if auto_carla_mode:
                carla_updates = _run_carla_validation_with_auto_instances(
                    sp=sp,
                    candidates=candidates,
                    worker_count=int(args.carla_auto_workers),
                    carla_host=str(args.carla_auto_host),
                    carla_root=args.carla_root,
                    carla_args=list(args.carla_arg or []),
                    carla_auto_port_min=int(args.carla_auto_port_min),
                    carla_auto_port_max=int(args.carla_auto_port_max),
                    carla_connect_retries=int(args.carla_auto_connect_retries),
                    carla_start_wait_s=float(args.carla_auto_start_wait),
                    carla_rpc_probe_timeout_s=float(args.carla_auto_rpc_probe_timeout),
                    carla_rpc_probe_retries=max(1, int(args.carla_auto_rpc_probe_retries)),
                    carla_repair_max_attempts=int(args.carla_repair_max_attempts),
                    carla_repair_xy_offsets=str(args.carla_repair_xy_offsets),
                    carla_repair_z_offsets=str(args.carla_repair_z_offsets),
                    carla_align_before_validate=bool(args.carla_align_before_validate),
                    carla_require_risk=bool(args.carla_require_risk),
                    carla_validation_timeout=float(args.carla_validation_timeout),
                    resume_state_path=resume_state_path if resume_enabled else None,
                    resume_completed_runs=resume_completed_runs if resume_enabled else None,
                )
            else:
                carla_updates = _run_carla_validation_on_instances(
                    sp=sp,
                    candidates=candidates,
                    instances=parsed_instances,
                    carla_repair_max_attempts=int(args.carla_repair_max_attempts),
                    carla_repair_xy_offsets=str(args.carla_repair_xy_offsets),
                    carla_repair_z_offsets=str(args.carla_repair_z_offsets),
                    carla_align_before_validate=bool(args.carla_align_before_validate),
                    carla_require_risk=bool(args.carla_require_risk),
                    carla_validation_timeout=float(args.carla_validation_timeout),
                    resume_state_path=resume_state_path if resume_enabled else None,
                    resume_completed_runs=resume_completed_runs if resume_enabled else None,
                )

        if bool(args.build_dashboard) and sp is not None:
            dashboard_info = sp._build_integrated_schema_dashboard(
                run_dirs=selected_run_dirs,
                debug_root=root,
                output_path=args.dashboard_html,
                json_path=args.dashboard_json,
                title=args.dashboard_title,
                extra_globs=args.dashboard_glob,
            )

        # Re-read summary/dashboard metadata after CARLA + dashboard updates.
        dashboard_map = _load_dashboard_feature_map(root)
        multi_map = _load_multi_run_result_map(root)
        records = _build_records(selected_run_dirs, dashboard_map, multi_map)

    elif bool(args.build_dashboard):
        repo_root = Path(__file__).resolve().parents[1]
        sp = _load_start_pipeline_module(repo_root)
        dashboard_info = sp._build_integrated_schema_dashboard(
            run_dirs=selected_run_dirs,
            debug_root=root,
            output_path=args.dashboard_html,
            json_path=args.dashboard_json,
            title=args.dashboard_title,
            extra_globs=args.dashboard_glob,
        )
        dashboard_map = _load_dashboard_feature_map(root)
        multi_map = _load_multi_run_result_map(root)
        records = _build_records(selected_run_dirs, dashboard_map, multi_map)

    aggregate = _summarize(records, ignore_carla_fails=bool(args.ignore_carla_fails))
    aggregate["root"] = str(root)
    aggregate["requested_limit"] = int(args.limit)
    aggregate["total_available_runs"] = len(run_dirs)
    aggregate["analyzed_runs"] = len(records)
    aggregate["records"] = records
    aggregate["carla_validation_run"] = {
        "requested": bool(run_carla),
        "mode": (
            "auto_start"
            if bool(auto_carla_mode and run_carla)
            else ("external_instances" if bool(run_carla) else "disabled")
        ),
        "instances": [f"{h}:{p}" for h, p in parsed_instances],
        "auto_start": {
            "enabled": bool(auto_carla_mode and run_carla),
            "workers": int(args.carla_auto_workers) if auto_carla_mode and run_carla else None,
            "host": str(args.carla_auto_host) if auto_carla_mode and run_carla else None,
            "port_min": int(args.carla_auto_port_min) if auto_carla_mode and run_carla else None,
            "port_max": int(args.carla_auto_port_max) if auto_carla_mode and run_carla else None,
            "connect_retries": int(args.carla_auto_connect_retries) if auto_carla_mode and run_carla else None,
            "launch_wait_s": float(args.carla_auto_start_wait) if auto_carla_mode and run_carla else None,
            "rpc_probe_timeout_s": float(args.carla_auto_rpc_probe_timeout) if auto_carla_mode and run_carla else None,
            "rpc_probe_retries": int(args.carla_auto_rpc_probe_retries) if auto_carla_mode and run_carla else None,
            "carla_root": (
                str(Path(args.carla_root).resolve())
                if (auto_carla_mode and run_carla and args.carla_root is not None)
                else None
            ),
            "carla_args": list(args.carla_arg or []) if auto_carla_mode and run_carla else [],
        },
        "updated_runs": len(carla_updates),
        "resume": {
            "enabled": bool(resume_enabled),
            "state_path": str(resume_state_path) if resume_state_path is not None else None,
            "resume_reset": bool(args.carla_resume_reset) if run_carla else False,
            "initial_completed_runs": int(len(resume_completed_runs)),
        },
    }
    aggregate["dashboard"] = dashboard_info

    _print_report(
        aggregate,
        records,
        int(args.limit),
        root,
        int(args.top_failures),
        bool(args.ignore_carla_fails),
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
