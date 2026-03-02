#!/usr/bin/env python3
"""Batch-run `v2xpnp.pipeline.entrypoint` with parallel workers and per-route timeout.

Features:
- Runs multiple scenario directories concurrently.
- Terminates any scenario run that exceeds timeout.
- Captures per-scenario stdout/stderr into a log file.
- Prints a final summary of successes/failures.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence


DEFAULT_SCENARIOS: List[str] = [
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-14-04-53_21_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-14-07-53_24_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-14-29-53_46_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-14-33-53_50_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-14-34-53_51_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-14-35-53_52_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-39-17_11_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-39-17_11_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-41-17_13_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-41-17_13_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-42-18_14_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-47-17_19_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-51-17_23_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-58-18_30_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-15-58-18_30_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-16-06-18_38_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-04-16-12-17_44_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-14-29-45_3_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-14-34-45_8_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-15-11-13_2_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-16-10-26_7_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-16-16-26_13_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-16-17-26_14_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-16-18-26_15_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-16-24-26_21_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-16-25-26_22_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-05-16-28-26_25_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-07-14-41-37_11_1",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-07-15-01-15_0_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-07-15-08-15_7_0",
    "/data2/marco/CoLMDriver/v2xpnp/dataset/2023-04-07-15-08-15_7_2",
]


@dataclass
class RunResult:
    scenario: str
    status: str
    elapsed_s: float
    return_code: int | None
    log_path: str
    message: str = ""


def _read_scenarios_from_file(path: Path) -> List[str]:
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5.0)
        return
    except Exception:
        pass
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _run_one(
    scenario_dir: Path,
    timeout_s: float,
    workspace_root: Path,
    python_exec: str,
    extra_args: Sequence[str],
    log_name: str,
) -> RunResult:
    scenario_str = str(scenario_dir)
    log_path = scenario_dir / log_name
    html_path = scenario_dir / "trajectory_plot.html"
    start = time.perf_counter()

    if not scenario_dir.is_dir():
        return RunResult(
            scenario=scenario_str,
            status="missing_dir",
            elapsed_s=0.0,
            return_code=None,
            log_path=str(log_path),
            message="Scenario directory does not exist",
        )

    cmd = [python_exec, "-m", "v2xpnp.pipeline.entrypoint", scenario_str]
    if extra_args:
        cmd.extend(list(extra_args))

    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"[BATCH] cmd: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(workspace_root),
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=float(timeout_s))
            elapsed = time.perf_counter() - start
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            elapsed = time.perf_counter() - start
            return RunResult(
                scenario=scenario_str,
                status="timeout",
                elapsed_s=elapsed,
                return_code=None,
                log_path=str(log_path),
                message=f"Exceeded timeout ({timeout_s:.0f}s)",
            )

    if rc != 0:
        return RunResult(
            scenario=scenario_str,
            status="nonzero_exit",
            elapsed_s=elapsed,
            return_code=int(rc),
            log_path=str(log_path),
            message=f"Entrypoint exited with code {rc}",
        )

    if not html_path.exists():
        return RunResult(
            scenario=scenario_str,
            status="missing_output",
            elapsed_s=elapsed,
            return_code=0,
            log_path=str(log_path),
            message="No trajectory_plot.html generated",
        )

    return RunResult(
        scenario=scenario_str,
        status="ok",
        elapsed_s=elapsed,
        return_code=0,
        log_path=str(log_path),
        message="",
    )


def _print_result_line(idx: int, total: int, rr: RunResult) -> None:
    base = f"[{idx:02d}/{total:02d}] {Path(rr.scenario).name} -> {rr.status} ({rr.elapsed_s:.1f}s)"
    if rr.message:
        base += f" | {rr.message}"
    print(base, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run v2xpnp.pipeline.entrypoint with parallel workers and timeout.",
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="Scenario directories. If empty, uses the built-in default list.",
    )
    parser.add_argument(
        "--scenarios-file",
        type=Path,
        default=None,
        help="Optional text file (one scenario path per line).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-scenario timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/pipeline_batch_summary.json"),
        help="Where to save JSON summary.",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="trajectory_plot.batch_run.log",
        help="Per-scenario log filename (default: trajectory_plot.batch_run.log).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to v2xpnp.pipeline.entrypoint.",
    )
    args = parser.parse_args()

    scenarios: List[str] = []
    if args.scenarios_file is not None:
        scenarios.extend(_read_scenarios_from_file(args.scenarios_file))
    if args.scenarios:
        scenarios.extend(args.scenarios)
    if not scenarios:
        scenarios = list(DEFAULT_SCENARIOS)

    scenario_paths = [Path(s).expanduser().resolve() for s in scenarios]
    timeout_s = float(args.timeout)
    workers = max(1, int(args.workers))
    summary_json = args.summary_json.expanduser().resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    workspace_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable
    extra_args: List[str] = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    print(f"[BATCH] scenarios={len(scenario_paths)} workers={workers} timeout={timeout_s:.0f}s", flush=True)
    if extra_args:
        print(f"[BATCH] forwarding extra args: {extra_args}", flush=True)

    start_all = time.perf_counter()
    results: List[RunResult] = []
    submitted = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_scenario = {}
        for sp in scenario_paths:
            submitted += 1
            fut = ex.submit(
                _run_one,
                sp,
                timeout_s,
                workspace_root,
                python_exec,
                extra_args,
                str(args.log_name),
            )
            fut_to_scenario[fut] = sp

        for fut in as_completed(fut_to_scenario):
            completed += 1
            rr = fut.result()
            results.append(rr)
            _print_result_line(completed, submitted, rr)

    # Keep output deterministic
    results.sort(key=lambda r: r.scenario)
    failed = [r for r in results if r.status != "ok"]

    payload = {
        "started_at_epoch_s": time.time(),
        "elapsed_total_s": time.perf_counter() - start_all,
        "workers": workers,
        "timeout_s": timeout_s,
        "total": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "failed": len(failed),
        "results": [asdict(r) for r in results],
        "failed_scenarios": [r.scenario for r in failed],
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("", flush=True)
    print("[BATCH] Summary", flush=True)
    print(f"  total={payload['total']} ok={payload['ok']} failed={payload['failed']}", flush=True)
    print(f"  summary_json={summary_json}", flush=True)
    if failed:
        print("  failed routes:", flush=True)
        for rr in failed:
            name = Path(rr.scenario).name
            detail = rr.message or rr.status
            print(f"    - {name}: {rr.status} ({rr.elapsed_s:.1f}s) | {detail}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

