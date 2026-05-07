"""HITL dashboard backend.

Implements the routes described in `docs/hitl_colmdriver_plan.md` §4.3.
The CoLMDriver agent calls `/api/negotiate` from the team_code-side
HitlDecisionProvider (Phase 3); browser clients consume the rest.

Run:
    pip install fastapi 'uvicorn[standard]'
    python -m tools.hitl_dashboard --port 8765

Forward over SSH from your laptop:
    ssh -L 8765:localhost:8765 <eval-server>
    # then open http://localhost:8765

Optional decision log:
    python -m tools.hitl_dashboard --port 8765 \
        --decision-log runs/hitl_decisions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover
    print(
        "[hitl_dashboard] FastAPI not installed. Run: "
        "pip install fastapi 'uvicorn[standard]'",
        file=sys.stderr,
    )
    raise

from tools.hitl_dashboard.vocab import NAV, SPEED, validate_decision


# ---------------------------------------------------------------------------
# In-memory state. The agent will eventually populate this via internal
# helpers (Phase 3); for now the backend exposes test-only setters.
# ---------------------------------------------------------------------------


@dataclass
class EgoState:
    ego_id: int
    mode: str = "autopilot"  # "autopilot" | "negotiating" | "direct"
    last_speed_cmd: Optional[str] = None
    last_nav_cmd: Optional[str] = None
    measurements: Dict[str, Any] = field(default_factory=dict)
    # Terminal lifecycle. None = active. Otherwise one of:
    #   "goal_reached"        — auto-set when telemetry reports completed=True
    #   "actor_destroyed"     — auto-set when telemetry reports actor_alive=False
    #   "operator_terminated" — set by POST /api/terminate_ego
    terminal_state: Optional[str] = None
    terminal_reason: Optional[str] = None
    terminal_at_step: Optional[int] = None
    terminal_at_time: Optional[float] = None


@dataclass
class PendingNegotiation:
    """A negotiation awaiting human input. One at a time (synchronous tick)."""
    request_id: str
    scenario_id: str
    step: int
    timestamp_s: float
    group: List[int]
    comm_info_list: List[Dict[str, Any]]
    deadline_monotonic_s: float
    future: asyncio.Future  # resolved with {ego_id: {"speed": ..., "nav": ...}}


@dataclass
class DecisionLogEntry:
    timestamp_s: float
    scenario_id: str
    step: int
    ego_id: int
    mode: str
    speed: str
    nav: str
    source: str  # "human" | "timeout-fallback"


class HitlState:
    def __init__(self) -> None:
        self.scenario_id: str = ""
        self.step: int = 0
        self.egos: Dict[int, EgoState] = {}
        self.pending: Optional[PendingNegotiation] = None
        self.decision_log: Deque[DecisionLogEntry] = deque(maxlen=10_000)
        self._lock = asyncio.Lock()
        self._ws_clients: List[WebSocket] = []
        # Optional: append-only JSONL decision sink. Set via --decision-log
        # CLI flag or HITL_DECISION_LOG env var. None = no file logging.
        self.decision_log_path: Optional[Path] = None
        # Hold-last decisions for direct mode: per-ego latest {speed, nav}.
        # The agent polls /api/current_decisions every tick; the operator
        # updates via /api/set_decision. Default: KEEP / follow the lane.
        self.current_decisions: Dict[int, Dict[str, str]] = {}
        # Bumps every time current_decisions is mutated so the agent (or any
        # poller) can short-circuit if nothing changed since last fetch.
        self.current_decisions_version: int = 0
        # Latest JPEG bytes per (ego_id, view). View is "front" by default;
        # multi-camera builds will populate "rear" / "left" / "right".
        # Stored in memory; never persisted.
        self.frames: Dict[str, bytes] = {}
        self.frame_versions: Dict[str, int] = {}
        # Per-ego route polyline + goal — sent by the agent at registration
        # time and surfaced to the frontend mini-map via /api/route.
        self.routes: Dict[int, Dict[str, Any]] = {}
        # Pending one-shot Return-to-route requests. The operator clicks
        # the button → /api/return_to_route sets the flag here → the
        # agent picks it up via /api/current_decisions on its next poll
        # → the agent runs the merge and clears the flag back.
        self.return_to_route_pending: Dict[int, bool] = {}
        # Multi-scenario queue position for the top-bar progress label.
        self.queue_index: int = 0
        self.queue_total: int = 0
        # Operator clicked "End scenario" — hitl_run polls this flag in
        # its eval-wait loop and gracefully terminates the eval child
        # so the queue advances to the next scenario.
        self.end_scenario_pending: bool = False
        # Operator clicked "Retry scenario" — hitl_run polls this flag and
        # restarts the SAME scenario from scratch (same idea as
        # end_scenario_pending, but the launcher does not advance the
        # queue index and wipes the per-scenario results dir before
        # respawning the eval). Used when the human messes up and wants a
        # clean redo.
        self.retry_scenario_pending: bool = False
        # Results-dir bundle: append-only logs + a live snapshot. All
        # filenames derived from a single root via `_set_results_dir`.
        self.results_dir: Optional[Path] = None
        self.telemetry_log_path: Optional[Path] = None
        self.terminal_log_path: Optional[Path] = None
        self.results_json_path: Optional[Path] = None
        self.trajectory_csv_path: Optional[Path] = None
        self._trajectory_csv_header_written: bool = False
        self.run_started_at: float = time.time()
        self.run_started_iso: str = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._last_snapshot_step: int = -1
        # Ready-gate: agent waits for the operator's browser to connect
        # before it starts driving. Flips True the first time a WS client
        # is accepted; once True, stays True even if every browser closes.
        self.ready: bool = False
        self.first_ready_at: Optional[float] = None
        self.ws_clients_seen: int = 0

    def set_results_dir(self, root: Optional[Path]) -> None:
        if root is None:
            self.results_dir = None
            self.telemetry_log_path = None
            self.terminal_log_path = None
            self.results_json_path = None
            self.trajectory_csv_path = None
            self._trajectory_csv_header_written = False
            return
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.results_dir = root
        self.telemetry_log_path = root / "telemetry.jsonl"
        self.terminal_log_path = root / "terminal_events.jsonl"
        self.results_json_path = root / "results.json"
        # Analysis-ready flat CSV: one row per ego per tick. Pandas/excel-
        # friendly. Header written on first row.
        self.trajectory_csv_path = root / "trajectory.csv"
        self._trajectory_csv_header_written = self.trajectory_csv_path.exists()
        # decision_log_path is set separately (CLI may override its name);
        # default it inside the results dir if the operator didn't override.
        if self.decision_log_path is None:
            self.decision_log_path = root / "decisions.jsonl"

    # --- public API used by the agent (Phase 3) and tests -----------------

    async def upsert_ego(self, ego: EgoState) -> None:
        async with self._lock:
            self.egos[ego.ego_id] = ego
        await self._broadcast({"type": "ego_update", "ego_id": ego.ego_id})

    async def set_step(self, scenario_id: str, step: int) -> None:
        async with self._lock:
            self.scenario_id = scenario_id
            self.step = step
        await self._broadcast({"type": "tick", "scenario_id": scenario_id, "step": step})

    async def open_negotiation(
        self,
        *,
        scenario_id: str,
        step: int,
        group: List[int],
        comm_info_list: List[Dict[str, Any]],
        timeout_s: float,
    ) -> Dict[int, Dict[str, str]]:
        """Block until the operator submits, or timeout — then return decisions.

        On timeout, fills in DEFAULT_FALLBACK_{SPEED,NAV} for every ego in the
        group and logs a warning. The agent never sees an exception here.
        """
        from tools.hitl_dashboard.vocab import DEFAULT_FALLBACK_NAV, DEFAULT_FALLBACK_SPEED

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        request_id = f"{scenario_id}:{step}:{int(time.monotonic() * 1000)}"
        deadline = time.monotonic() + max(timeout_s, 1.0)

        async with self._lock:
            if self.pending is not None:
                raise RuntimeError(
                    "another negotiation is already pending; this should not "
                    "happen with the synchronous CoLMDriver tick model"
                )
            self.pending = PendingNegotiation(
                request_id=request_id,
                scenario_id=scenario_id,
                step=step,
                timestamp_s=time.time(),
                group=list(group),
                comm_info_list=list(comm_info_list),
                deadline_monotonic_s=deadline,
                future=future,
            )
            for ego_id in group:
                self.egos.setdefault(ego_id, EgoState(ego_id=ego_id)).mode = "negotiating"

        await self._broadcast({"type": "negotiation_open", "request_id": request_id})

        source = "human"
        try:
            decisions: Dict[int, Dict[str, str]] = await asyncio.wait_for(
                future, timeout=timeout_s
            )
        except asyncio.TimeoutError:
            source = "timeout-fallback"
            decisions = {
                ego_id: {"speed": DEFAULT_FALLBACK_SPEED, "nav": DEFAULT_FALLBACK_NAV}
                for ego_id in group
            }
            print(
                f"[hitl_dashboard] WARN timeout waiting for human input "
                f"(scenario={scenario_id}, step={step}, group={group}); "
                f"applying KEEP+follow-the-lane fallback."
            )

        # Log + flip egos back to autopilot.
        new_entries: List[DecisionLogEntry] = []
        async with self._lock:
            self.pending = None
            for ego_id, dec in decisions.items():
                entry = DecisionLogEntry(
                    timestamp_s=time.time(),
                    scenario_id=scenario_id,
                    step=step,
                    ego_id=ego_id,
                    mode="negotiating",
                    speed=dec["speed"],
                    nav=dec["nav"],
                    source=source,
                )
                self.decision_log.append(entry)
                new_entries.append(entry)
                ego_state = self.egos.setdefault(ego_id, EgoState(ego_id=ego_id))
                ego_state.mode = "autopilot"
                ego_state.last_speed_cmd = dec["speed"]
                ego_state.last_nav_cmd = dec["nav"]

        # JSONL sink (best-effort, never block the agent on file I/O issues).
        if self.decision_log_path is not None and new_entries:
            try:
                self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.decision_log_path.open("a") as fh:
                    for entry in new_entries:
                        fh.write(json.dumps({
                            "timestamp_s": entry.timestamp_s,
                            "scenario_id": entry.scenario_id,
                            "step": entry.step,
                            "ego_id": entry.ego_id,
                            "mode": entry.mode,
                            "speed": entry.speed,
                            "nav": entry.nav,
                            "source": entry.source,
                        }) + "\n")
            except OSError as exc:
                print(
                    f"[hitl_dashboard] WARN failed to append decision log "
                    f"to {self.decision_log_path}: {exc}",
                    file=sys.stderr,
                )

        await self._broadcast({"type": "negotiation_resolved", "source": source})
        return decisions

    # --- websocket plumbing -----------------------------------------------

    async def attach_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        self._ws_clients.append(ws)
        self.ws_clients_seen += 1
        # NOTE: ready does NOT flip here. The dashboard might render slowly
        # or the operator may want to preview the cameras / route before
        # the egos start moving. The frontend's Start button posts to
        # /api/start when the operator is ready.

    def detach_ws(self, ws: WebSocket) -> None:
        try:
            self._ws_clients.remove(ws)
        except ValueError:
            pass

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        payload = json.dumps(message)
        for ws in self._ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.detach_ws(ws)


STATE = HitlState()


def _append_jsonl(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError as exc:
        print(f"[hitl_dashboard] WARN failed to append {path}: {exc}", file=sys.stderr)


# Flat schema for trajectory.csv. Order matters — written exactly once as
# the header row. Anything the agent doesn't supply is left blank.
_TRAJ_COLUMNS = (
    "wall_iso", "wall_ts", "scenario_id", "step", "sim_time_s",
    "ego_id",
    "x", "y", "z",
    "yaw_deg", "pitch_deg", "roll_deg",
    "vx", "vy", "vz",
    "speed_mps", "speed_kmh",
    "throttle", "brake", "steer", "hand_brake", "reverse",
    "applied_speed_cmd", "applied_nav_cmd",
    "distance_to_goal_m", "route_progress",
    "waypoints_remaining", "waypoints_total",
    "near_goal", "completed",
    "ds_pct", "rc_pct", "score_penalty",
    "efficiency_pct", "eta_sim_s",
    "scores_provenance",
    "off_route", "off_route_distance_m", "route_replaced",
    "actor_alive",
)


def _csv_escape(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return ""
        return repr(v) if isinstance(v, float) else str(v)
    s = str(v)
    if any(c in s for c in (",", '"', "\n", "\r")):
        return '"' + s.replace('"', '""') + '"'
    return s


def _append_trajectory_rows(scenario_id: str, step: int, sim_time_s: Optional[float],
                            egos: Dict[str, Any]) -> None:
    """Append one row per ego to <results-dir>/trajectory.csv. Header is
    written once on first call per file."""
    path = STATE.trajectory_csv_path
    if path is None or not egos:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        wall_ts = time.time()
        wall_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        rows_out = []
        if not STATE._trajectory_csv_header_written:
            rows_out.append(",".join(_TRAJ_COLUMNS))
            STATE._trajectory_csv_header_written = True
        for k, v in egos.items():
            if not isinstance(v, dict):
                continue
            row = {
                "wall_iso": wall_iso,
                "wall_ts": wall_ts,
                "scenario_id": scenario_id,
                "step": step,
                "sim_time_s": sim_time_s,
                "ego_id": k,
            }
            row.update(v)
            rows_out.append(",".join(_csv_escape(row.get(c)) for c in _TRAJ_COLUMNS))
        with path.open("a") as fh:
            fh.write("\n".join(rows_out) + "\n")
    except OSError as exc:
        print(f"[hitl_dashboard] WARN failed to append {path}: {exc}", file=sys.stderr)


def _build_results_snapshot() -> Dict[str, Any]:
    """One-shot picture of the run for results.json + GET /api/results."""
    egos_payload: Dict[str, Any] = {}
    aggregate_inf: Dict[str, int] = {}
    for ego_id, ego_state in sorted(STATE.egos.items()):
        m = dict(ego_state.measurements)
        # Roll up infractions for the global summary.
        for k, v in (m.get("infractions") or {}).items():
            try:
                aggregate_inf[k] = aggregate_inf.get(k, 0) + int(v)
            except Exception:
                pass
        egos_payload[str(ego_id)] = {
            "ego_id": ego_id,
            "mode": ego_state.mode,
            "last_speed_cmd": ego_state.last_speed_cmd,
            "last_nav_cmd": ego_state.last_nav_cmd,
            "terminal_state": ego_state.terminal_state,
            "terminal_reason": ego_state.terminal_reason,
            "terminal_at_step": ego_state.terminal_at_step,
            "terminal_at_time": ego_state.terminal_at_time,
            # Latest telemetry — the *last seen* values, which act as final
            # state once the ego goes terminal.
            "ds_pct": m.get("ds_pct"),
            "rc_pct": m.get("rc_pct"),
            "score_penalty": m.get("score_penalty"),
            "scores_provenance": m.get("scores_provenance"),
            "speed_kmh": m.get("speed_kmh"),
            "speed_mps": m.get("speed_mps"),
            "distance_to_goal_m": m.get("distance_to_goal_m"),
            "route_progress": m.get("route_progress"),
            "waypoints_remaining": m.get("waypoints_remaining"),
            "waypoints_total": m.get("waypoints_total"),
            "completed": m.get("completed"),
            "near_goal": m.get("near_goal"),
            "x": m.get("x"),
            "y": m.get("y"),
            "yaw_deg": m.get("yaw_deg"),
            "infractions": m.get("infractions") or {},
        }
    return {
        "scenario_id": STATE.scenario_id,
        "step": STATE.step,
        "sim_time_s": next(
            (e.measurements.get("sim_time_s") for e in STATE.egos.values()
             if e.measurements.get("sim_time_s") is not None),
            None,
        ),
        "started_at_unix": STATE.run_started_at,
        "started_at_iso": STATE.run_started_iso,
        "snapshot_at_unix": time.time(),
        "snapshot_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "egos": egos_payload,
        "infractions_aggregate": aggregate_inf,
        "decisions_logged": len(STATE.decision_log),
        "paths": {
            "results_dir": str(STATE.results_dir) if STATE.results_dir else None,
            "decision_log": str(STATE.decision_log_path) if STATE.decision_log_path else None,
            "telemetry_log": str(STATE.telemetry_log_path) if STATE.telemetry_log_path else None,
            "terminal_log": str(STATE.terminal_log_path) if STATE.terminal_log_path else None,
            "results_json": str(STATE.results_json_path) if STATE.results_json_path else None,
        },
    }


def _persist_snapshot() -> None:
    """Best-effort write of results.json. Never raises."""
    path = STATE.results_json_path
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = _build_results_snapshot()
        # Atomic write: serialize to a sibling .tmp then rename.
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(snapshot, fh, default=str, indent=2)
        tmp.replace(path)
    except OSError as exc:
        print(f"[hitl_dashboard] WARN failed to write {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# FastAPI app + routes
# ---------------------------------------------------------------------------

app = FastAPI(title="CoLMDriver HITL Dashboard")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_html = _STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(index_html.read_text())
    return HTMLResponse("<h1>HITL Dashboard</h1><p>static/index.html missing.</p>")


@app.get("/api/vocab")
async def get_vocab() -> Dict[str, List[str]]:
    return {"speed": list(SPEED), "nav": list(NAV)}


@app.get("/api/state")
async def get_state() -> Dict[str, Any]:
    return {
        "scenario_id": STATE.scenario_id,
        "step": STATE.step,
        "queue_index": STATE.queue_index,
        "queue_total": STATE.queue_total,
        "end_scenario_pending": STATE.end_scenario_pending,
        "retry_scenario_pending": STATE.retry_scenario_pending,
        "egos": [
            {
                "ego_id": e.ego_id,
                "mode": e.mode,
                "last_speed_cmd": e.last_speed_cmd,
                "last_nav_cmd": e.last_nav_cmd,
                "measurements": e.measurements,
            }
            for e in sorted(STATE.egos.values(), key=lambda x: x.ego_id)
        ],
    }


@app.get("/api/pending")
async def get_pending() -> Dict[str, Any]:
    p = STATE.pending
    if p is None:
        return {"pending": False}
    remaining = max(0.0, p.deadline_monotonic_s - time.monotonic())
    return {
        "pending": True,
        "request_id": p.request_id,
        "scenario_id": p.scenario_id,
        "step": p.step,
        "group": p.group,
        "comm_info_list": p.comm_info_list,
        "deadline_remaining_s": remaining,
    }


@app.post("/api/submit")
async def submit_decisions(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = STATE.pending
    if p is None:
        raise HTTPException(409, "no negotiation pending")

    request_id = payload.get("request_id")
    if request_id != p.request_id:
        raise HTTPException(409, f"stale request_id {request_id!r}; current={p.request_id!r}")

    raw = payload.get("decisions") or {}
    if not isinstance(raw, dict):
        raise HTTPException(400, "decisions must be {ego_id: {speed, nav}}")

    decisions: Dict[int, Dict[str, str]] = {}
    expected = set(p.group)
    for k, v in raw.items():
        try:
            ego_id = int(k)
        except (TypeError, ValueError):
            raise HTTPException(400, f"non-integer ego_id key: {k!r}")
        if ego_id not in expected:
            raise HTTPException(400, f"ego_id {ego_id} not in pending group {p.group}")
        validate_decision(v)
        decisions[ego_id] = {"speed": v["speed"], "nav": v["nav"]}

    missing = expected - set(decisions)
    if missing:
        raise HTTPException(400, f"missing decisions for ego_id(s): {sorted(missing)}")

    if not p.future.done():
        p.future.set_result(decisions)
    return {"ok": True, "applied": len(decisions)}


# --- Hold-last decisions (direct/always-on HITL mode) ----------------------
#
# In direct mode the agent does NOT block on /api/negotiate every tick.
# Instead it polls /api/current_decisions at high frequency and applies
# whatever the operator most recently submitted via /api/set_decision.
# Default decision = KEEP / follow the lane.


@app.get("/api/current_decisions")
async def get_current_decisions() -> Dict[str, Any]:
    """Latest hold-last decisions + terminal flags. Agent polls this every
    tick; the version bumps on either decision changes or terminal-state
    changes so the agent can short-circuit when nothing's new."""
    terminal: Dict[str, Dict[str, Any]] = {}
    for ego_id, ego_state in STATE.egos.items():
        if ego_state.terminal_state is None:
            continue
        terminal[str(ego_id)] = {
            "state": ego_state.terminal_state,
            "reason": ego_state.terminal_reason,
            "at_step": ego_state.terminal_at_step,
            "at_time": ego_state.terminal_at_time,
        }
    return {
        "version": STATE.current_decisions_version,
        "decisions": {str(k): v for k, v in STATE.current_decisions.items()},
        "terminal": terminal,
        "ready": STATE.ready,
        "first_ready_at": STATE.first_ready_at,
        "return_to_route_pending": {
            str(k): True for k in STATE.return_to_route_pending.keys()
        },
    }


@app.post("/api/terminate_ego")
async def terminate_ego(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Operator marks an ego terminal. Sticky until /api/clear_terminal."""
    try:
        ego_id = int(payload["ego_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "ego_id is required")
    reason = str(payload.get("reason") or "operator-terminated")
    async with STATE._lock:
        ego_state = STATE.egos.setdefault(ego_id, EgoState(ego_id=ego_id))
        ego_state.terminal_state = "operator_terminated"
        ego_state.terminal_reason = reason
        ego_state.terminal_at_step = STATE.step
        ego_state.terminal_at_time = time.time()
        STATE.current_decisions_version += 1
    _append_jsonl(STATE.terminal_log_path, {
        "ts": time.time(),
        "ego_id": ego_id,
        "state": "operator_terminated",
        "reason": reason,
        "at_step": STATE.step,
        "at_time": time.time(),
        "source": "operator",
    })
    _persist_snapshot()
    await STATE._broadcast({"type": "terminal_set", "ego_id": ego_id})
    return {"ok": True, "ego_id": ego_id, "state": "operator_terminated"}


@app.post("/api/clear_terminal")
async def clear_terminal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Undo a terminal flag. Only allowed for operator-set ones; auto-set
    states (goal_reached / actor_destroyed) are left alone — those reflect
    real ground truth from the simulator."""
    try:
        ego_id = int(payload["ego_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "ego_id is required")
    force = bool(payload.get("force", False))
    async with STATE._lock:
        ego_state = STATE.egos.get(ego_id)
        if ego_state is None or ego_state.terminal_state is None:
            return {"ok": True, "cleared": False}
        if ego_state.terminal_state != "operator_terminated" and not force:
            raise HTTPException(
                409,
                f"ego {ego_id} is in terminal state {ego_state.terminal_state!r}; "
                "auto-set states are sticky. Pass force=true to override.",
            )
        prev_state = ego_state.terminal_state
        ego_state.terminal_state = None
        ego_state.terminal_reason = None
        ego_state.terminal_at_step = None
        ego_state.terminal_at_time = None
        STATE.current_decisions_version += 1
    _append_jsonl(STATE.terminal_log_path, {
        "ts": time.time(),
        "ego_id": ego_id,
        "state": "cleared",
        "reason": f"cleared from {prev_state!r}",
        "at_step": STATE.step,
        "at_time": time.time(),
        "source": "operator",
        "force": force,
    })
    _persist_snapshot()
    await STATE._broadcast({"type": "terminal_cleared", "ego_id": ego_id})
    return {"ok": True, "cleared": True}


@app.post("/api/set_decision")
async def set_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Operator sets / replaces the held decision for one or more egos.

    Body shapes accepted:

        # single ego
        {"ego_id": 0, "speed": "FASTER", "nav": "follow the lane"}

        # multi-ego
        {"decisions": {"0": {"speed":"...","nav":"..."}, "1": {...}}}

        # apply to all known egos:
        {"all": true, "speed": "STOP", "nav": "follow the lane"}
    """
    new_entries: List[DecisionLogEntry] = []
    async with STATE._lock:
        scenario_id = STATE.scenario_id or "direct"
        step = STATE.step
        if "decisions" in payload and isinstance(payload["decisions"], dict):
            updates = {int(k): v for k, v in payload["decisions"].items()}
        elif payload.get("all"):
            speed = str(payload.get("speed", ""))
            nav = str(payload.get("nav", ""))
            validate_decision({"speed": speed, "nav": nav})
            ids = sorted(STATE.egos.keys()) or [0]
            updates = {i: {"speed": speed, "nav": nav} for i in ids}
        else:
            try:
                ego_id = int(payload["ego_id"])
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, "ego_id is required for single-ego set")
            speed = str(payload.get("speed", ""))
            nav = str(payload.get("nav", ""))
            validate_decision({"speed": speed, "nav": nav})
            updates = {ego_id: {"speed": speed, "nav": nav}}

        for ego_id, dec in updates.items():
            validate_decision(dec)
            STATE.current_decisions[ego_id] = {"speed": dec["speed"], "nav": dec["nav"]}
            ego_state = STATE.egos.setdefault(ego_id, EgoState(ego_id=ego_id))
            ego_state.last_speed_cmd = dec["speed"]
            ego_state.last_nav_cmd = dec["nav"]
            entry = DecisionLogEntry(
                timestamp_s=time.time(),
                scenario_id=scenario_id,
                step=step,
                ego_id=ego_id,
                mode="direct",
                speed=dec["speed"],
                nav=dec["nav"],
                source="human",
            )
            STATE.decision_log.append(entry)
            new_entries.append(entry)
        STATE.current_decisions_version += 1

    # JSONL sink (best-effort).
    if STATE.decision_log_path is not None and new_entries:
        try:
            STATE.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
            with STATE.decision_log_path.open("a") as fh:
                for entry in new_entries:
                    fh.write(json.dumps({
                        "timestamp_s": entry.timestamp_s,
                        "scenario_id": entry.scenario_id,
                        "step": entry.step,
                        "ego_id": entry.ego_id,
                        "mode": entry.mode,
                        "speed": entry.speed,
                        "nav": entry.nav,
                        "source": entry.source,
                    }) + "\n")
        except OSError as exc:
            print(
                f"[hitl_dashboard] WARN failed to append decision log: {exc}",
                file=sys.stderr,
            )

    await STATE._broadcast({"type": "decision_set", "count": len(updates)})
    return {
        "ok": True,
        "applied": len(updates),
        "version": STATE.current_decisions_version,
    }


@app.post("/api/telemetry")
async def post_telemetry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Agent posts per-ego live state every tick.

    Body:
        {
          "scenario_id": "...",
          "step": 1234,
          "egos": {
            "0": {"speed_mps": 7.2, "x": 196.6, "y": 76.2, "yaw_deg": -38.4,
                  "distance_to_goal_m": 124.5, "route_progress": 0.05,
                  "near_goal": false, "completed": false, ...},
            ...
          }
        }
    """
    scenario_id = str(payload.get("scenario_id") or STATE.scenario_id or "direct")
    step = int(payload.get("step", STATE.step))
    egos = payload.get("egos") or {}
    if not isinstance(egos, dict):
        raise HTTPException(400, "egos must be a dict {ego_id_str: {...}}")
    promoted: List[int] = []
    async with STATE._lock:
        STATE.scenario_id = scenario_id
        STATE.step = step
        for k, v in egos.items():
            try:
                ego_id = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(v, dict):
                continue
            ego_state = STATE.egos.setdefault(ego_id, EgoState(ego_id=ego_id))
            ego_state.measurements = {kk: vv for kk, vv in v.items()}
            # Auto-promote terminal states from telemetry. Operator-set
            # terminations are sticky and never overwritten here.
            if ego_state.terminal_state == "operator_terminated":
                continue
            if v.get("actor_alive") is False:
                if ego_state.terminal_state != "actor_destroyed":
                    ego_state.terminal_state = "actor_destroyed"
                    ego_state.terminal_reason = "CARLA actor missing or destroyed"
                    ego_state.terminal_at_step = step
                    ego_state.terminal_at_time = time.time()
                    promoted.append(ego_id)
            elif v.get("completed"):
                if ego_state.terminal_state != "goal_reached":
                    ego_state.terminal_state = "goal_reached"
                    ego_state.terminal_reason = (
                        f"reached route goal "
                        f"(d={v.get('distance_to_goal_m'):.1f} m)"
                        if isinstance(v.get('distance_to_goal_m'), (int, float))
                        else "reached route goal"
                    )
                    ego_state.terminal_at_step = step
                    ego_state.terminal_at_time = time.time()
                    promoted.append(ego_id)
    # Append the raw telemetry frame to the JSONL stream — every tick is
    # captured so the run can be replayed/analyzed offline. Cheap (<2 KiB
    # per ego per tick).
    sim_time_s = payload.get("sim_time_s")
    _append_jsonl(STATE.telemetry_log_path, {
        "ts": time.time(),
        "scenario_id": scenario_id,
        "step": step,
        "sim_time_s": sim_time_s,
        "egos": egos,
    })
    # Flat per-ego rows for analysis. `trajectory.csv` is the
    # pandas/excel-friendly view of the telemetry stream.
    _append_trajectory_rows(scenario_id, step, sim_time_s, egos)

    # Auto-terminal promotions also go into the terminal-events stream so
    # results.json can be rebuilt from JSONLs alone.
    for ego_id in promoted:
        ego_state = STATE.egos.get(ego_id)
        if ego_state is None or ego_state.terminal_state is None:
            continue
        _append_jsonl(STATE.terminal_log_path, {
            "ts": time.time(),
            "ego_id": ego_id,
            "state": ego_state.terminal_state,
            "reason": ego_state.terminal_reason,
            "at_step": ego_state.terminal_at_step,
            "at_time": ego_state.terminal_at_time,
            "source": "auto",
        })

    # Snapshot results.json. Done after every telemetry tick — atomic
    # write, write rate is bounded by tick rate (~10 Hz), file is small.
    _persist_snapshot()

    if promoted:
        STATE.current_decisions_version += 1
        await STATE._broadcast({"type": "terminal_promoted", "ego_ids": promoted})
    return {"ok": True, "applied": len(egos), "promoted": promoted}


@app.post("/api/register_egos")
async def register_egos(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Agent calls this once at startup to declare the egos it controls.

    Body shape:
        {
          "scenario_id": "...",
          "ego_ids": [0, 1, 2],
          # Optional — polylines for the mini map.
          "routes": {
            "0": {"xy": [[x,y], ...], "goal": [x,y]},
            ...
          }
        }
    """
    scenario_id = str(payload.get("scenario_id", "direct"))
    ego_ids = [int(x) for x in (payload.get("ego_ids") or [])]
    routes_in = payload.get("routes") or {}
    async with STATE._lock:
        STATE.scenario_id = scenario_id
        for ego_id in ego_ids:
            if ego_id not in STATE.current_decisions:
                STATE.current_decisions[ego_id] = {
                    "speed": "KEEP",
                    "nav": "follow the lane",
                }
            STATE.egos.setdefault(ego_id, EgoState(ego_id=ego_id, mode="direct"))
        for k, v in routes_in.items():
            try:
                ego_id = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(v, dict):
                continue
            xy = v.get("xy")
            goal = v.get("goal")
            STATE.routes[ego_id] = {
                "xy": [[float(p[0]), float(p[1])] for p in (xy or [])
                       if isinstance(p, (list, tuple)) and len(p) >= 2],
                "goal": ([float(goal[0]), float(goal[1])]
                         if isinstance(goal, (list, tuple)) and len(goal) >= 2 else None),
            }
        STATE.current_decisions_version += 1
    await STATE._broadcast({"type": "egos_registered"})
    return {
        "ok": True,
        "scenario_id": scenario_id,
        "ego_ids": ego_ids,
        "version": STATE.current_decisions_version,
    }


@app.get("/api/route/{ego_id}")
async def get_route(ego_id: int) -> Dict[str, Any]:
    """Polyline + goal for the per-ego mini map. Empty if unknown."""
    r = STATE.routes.get(int(ego_id))
    if r is None:
        return {"xy": [], "goal": None}
    return {"xy": r.get("xy") or [], "goal": r.get("goal")}


@app.post("/api/return_to_route")
async def post_return_to_route(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Operator-initiated rejoin. Marks a pending flag the agent picks up
    on its next /api/current_decisions poll, AND auto-resets the held
    `nav` decision to 'follow the lane' so the operator can see the lane-
    change button de-highlight immediately (otherwise a stale 'right lane
    change' would stay highlighted while the merge was in flight)."""
    try:
        ego_id = int(payload["ego_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "ego_id is required")
    async with STATE._lock:
        STATE.return_to_route_pending[ego_id] = True
        # Auto-reset the held nav. The agent's lane-change action is one-
        # shot; once it's been triggered, leaving the nav at "left lane
        # change" misrepresents the current intent.
        cur = STATE.current_decisions.get(ego_id) or {"speed": "KEEP", "nav": "follow the lane"}
        STATE.current_decisions[ego_id] = {"speed": cur.get("speed", "KEEP"), "nav": "follow the lane"}
        STATE.current_decisions_version += 1
    await STATE._broadcast({"type": "return_to_route_requested", "ego_id": ego_id})
    return {"ok": True, "ego_id": ego_id, "pending": True}


@app.post("/api/results_dir")
async def post_results_dir(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Swap the active results directory mid-run. hitl_run calls this
    between scenarios so each scenario's bundle ends up in its own
    sub-folder under <experiment_id>/."""
    path = payload.get("path")
    if not path:
        raise HTTPException(400, "path is required")
    p = Path(str(path)).expanduser().resolve()
    async with STATE._lock:
        # Reset bookkeeping so the new bundle starts clean.
        STATE.run_started_at = time.time()
        STATE.run_started_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        STATE.set_results_dir(p)
        STATE.decision_log = STATE.decision_log  # noop, just satisfies linters
    return {
        "ok": True,
        "results_dir": str(STATE.results_dir),
        "telemetry_log": str(STATE.telemetry_log_path) if STATE.telemetry_log_path else None,
    }


@app.post("/api/scenario_started")
async def post_scenario_started(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reset per-ego state when a new scenario begins. Called by hitl_run
    between scenarios. Existing logs/results in the previous results dir
    stay on disk; in-memory state for cards, frames, decisions, terminals
    is wiped so the dashboard reflects only the current scenario.
    """
    scenario_id = str(payload.get("scenario_id") or "")
    queue_index = int(payload.get("queue_index", 0))
    queue_total = int(payload.get("queue_total", 0))
    async with STATE._lock:
        STATE.scenario_id = scenario_id
        STATE.step = 0
        STATE.egos = {}
        STATE.current_decisions = {}
        STATE.current_decisions_version += 1
        STATE.return_to_route_pending = {}
        STATE.routes = {}
        STATE.frames = {}
        STATE.frame_versions = {}
        STATE.pending = None
        # Each scenario waits for the operator to click Start again so
        # they can preview the new route/cameras before the egos move.
        STATE.ready = False
        STATE.first_ready_at = None
        # Reset the run timer so sim_time_s starts fresh.
        STATE.run_started_at = time.time()
        STATE.run_started_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        STATE.queue_index = queue_index
        STATE.queue_total = queue_total
        STATE.end_scenario_pending = False
        STATE.retry_scenario_pending = False
        # New scenario → fresh CSV header for this scenario's trajectory.csv.
        STATE._trajectory_csv_header_written = False
    await STATE._broadcast({
        "type": "scenario_started",
        "scenario_id": scenario_id,
        "queue_index": queue_index,
        "queue_total": queue_total,
    })
    return {"ok": True}


@app.post("/api/return_to_route/ack")
async def post_return_to_route_ack(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Agent calls this after acting on a Return-to-route request, with
    the result so the dashboard can log it / clear the pending flag."""
    try:
        ego_id = int(payload["ego_id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "ego_id is required")
    ok = bool(payload.get("ok", False))
    reason = str(payload.get("reason") or "")
    async with STATE._lock:
        STATE.return_to_route_pending.pop(ego_id, None)
        STATE.current_decisions_version += 1
    await STATE._broadcast({"type": "return_to_route_done", "ego_id": ego_id, "ok": ok, "reason": reason})
    return {"ok": True}


# --- Camera frames (live view) ---------------------------------------------
#
# Agent posts JPEG-encoded frames per ego. Browser renders them as <img>
# elements with a cache-busting query param.

_FRAME_VIEWS = ("front", "rear", "left", "right")


def _frame_key(ego_id: int, view: str) -> str:
    return f"{int(ego_id)}/{view}"


@app.post("/api/frame/{ego_id}/{view}")
async def post_frame(ego_id: int, view: str, request: Request) -> Dict[str, Any]:
    if view not in _FRAME_VIEWS:
        raise HTTPException(400, f"unknown view {view!r}; expected one of {_FRAME_VIEWS}")
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty frame body")
    if len(body) > 8 * 1024 * 1024:  # 8 MiB hard cap
        raise HTTPException(413, "frame too large (>8 MiB)")
    key = _frame_key(ego_id, view)
    STATE.frames[key] = body
    STATE.frame_versions[key] = STATE.frame_versions.get(key, 0) + 1
    return {"ok": True, "bytes": len(body), "version": STATE.frame_versions[key]}


@app.post("/api/frame/{ego_id}")
async def post_frame_default(ego_id: int, request: Request) -> Dict[str, Any]:
    return await post_frame(ego_id, "front", request)


@app.get("/api/frame/{ego_id}/{view}.jpg")
async def get_frame(ego_id: int, view: str) -> Response:
    if view not in _FRAME_VIEWS:
        raise HTTPException(404, "no such view")
    body = STATE.frames.get(_frame_key(ego_id, view))
    if body is None:
        # Tiny 1×1 transparent PNG keeps the <img> from showing a broken icon.
        empty = bytes.fromhex(
            "89504E470D0A1A0A0000000D4948445200000001000000010806000000"
            "1F15C4890000000A49444154789C6300010000000500010D0A2DB40000"
            "000049454E44AE426082"
        )
        return Response(content=empty, media_type="image/png")
    return Response(content=body, media_type="image/jpeg")


@app.get("/api/frame/{ego_id}.jpg")
async def get_frame_default(ego_id: int) -> Response:
    return await get_frame(ego_id, "front")


@app.get("/api/frames/versions")
async def get_frame_versions() -> Dict[str, Any]:
    """Returns per-(ego, view) frame version. Frontend polls this and
    only refreshes <img> elements whose version changed."""
    out: Dict[str, Dict[str, int]] = {}
    for key, ver in STATE.frame_versions.items():
        ego_str, view = key.split("/", 1)
        out.setdefault(ego_str, {})[view] = ver
    return {"versions": out}


@app.get("/api/ready")
async def get_ready() -> Dict[str, Any]:
    return {
        "ready": STATE.ready,
        "first_ready_at": STATE.first_ready_at,
        "ws_clients_seen": STATE.ws_clients_seen,
        "ws_clients_now": len(STATE._ws_clients),
    }


@app.post("/api/end_scenario")
async def post_end_scenario(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Operator finalizes the current scenario early.

    Marks every still-active ego as `operator_terminated`, writes the
    final results.json snapshot, and sets a pending flag that hitl_run
    polls — once set, hitl_run gracefully terminates the eval subprocess
    and advances to the next scenario in the queue.
    """
    payload = payload or {}
    reason = str(payload.get("reason") or "ended by operator")
    terminated: List[int] = []
    async with STATE._lock:
        STATE.end_scenario_pending = True
        for ego_id, ego_state in list(STATE.egos.items()):
            if ego_state.terminal_state is None:
                ego_state.terminal_state = "operator_terminated"
                ego_state.terminal_reason = reason
                ego_state.terminal_at_step = STATE.step
                ego_state.terminal_at_time = time.time()
                terminated.append(ego_id)
        STATE.current_decisions_version += 1
    for ego_id in terminated:
        _append_jsonl(STATE.terminal_log_path, {
            "ts": time.time(),
            "ego_id": ego_id,
            "state": "operator_terminated",
            "reason": reason,
            "at_step": STATE.step,
            "at_time": time.time(),
            "source": "end_scenario",
        })
    _persist_snapshot()
    await STATE._broadcast({"type": "scenario_end_requested", "terminated": terminated})
    return {
        "ok": True,
        "terminated_egos": terminated,
        "results_json": str(STATE.results_json_path) if STATE.results_json_path else None,
    }


@app.post("/api/retry_scenario")
async def post_retry_scenario(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Operator wants to redo the current scenario from scratch.

    Marks every still-active ego as `operator_terminated` (so the agent's
    direct-mode loop hard-brakes them while the launcher is killing the
    eval), and sets `retry_scenario_pending`. hitl_run polls the flag,
    terminates the eval child, wipes the per-scenario results dir, and
    respawns the SAME scenario without advancing the queue index.
    """
    payload = payload or {}
    reason = str(payload.get("reason") or "retry by operator")
    terminated: List[int] = []
    async with STATE._lock:
        STATE.retry_scenario_pending = True
        for ego_id, ego_state in list(STATE.egos.items()):
            if ego_state.terminal_state is None:
                ego_state.terminal_state = "operator_terminated"
                ego_state.terminal_reason = reason
                ego_state.terminal_at_step = STATE.step
                ego_state.terminal_at_time = time.time()
                terminated.append(ego_id)
        STATE.current_decisions_version += 1
    for ego_id in terminated:
        _append_jsonl(STATE.terminal_log_path, {
            "ts": time.time(),
            "ego_id": ego_id,
            "state": "operator_terminated",
            "reason": reason,
            "at_step": STATE.step,
            "at_time": time.time(),
            "source": "retry_scenario",
        })
    await STATE._broadcast({"type": "scenario_retry_requested", "terminated": terminated})
    return {
        "ok": True,
        "terminated_egos": terminated,
        "scenario_id": STATE.scenario_id,
    }


@app.post("/api/start")
async def post_start() -> Dict[str, Any]:
    """Operator clicks Start in the browser → flips ready=True. The agent
    is polling /api/current_decisions; on its next poll it sees ready
    and stops force-braking."""
    async with STATE._lock:
        if not STATE.ready:
            STATE.ready = True
            STATE.first_ready_at = time.time()
            STATE.current_decisions_version += 1
            print(f"[hitl_dashboard] READY — operator clicked Start; agent will begin driving")
    await STATE._broadcast({"type": "ready"})
    return {"ok": True, "ready": True, "first_ready_at": STATE.first_ready_at}


@app.get("/api/results")
async def get_results() -> Dict[str, Any]:
    """Live snapshot of the run — same payload that's written to results.json."""
    return _build_results_snapshot()


@app.get("/api/log")
async def get_log(limit: int = 200) -> Dict[str, Any]:
    n = max(1, min(int(limit), 10_000))
    items = list(STATE.decision_log)[-n:]
    return {
        "count": len(items),
        "entries": [
            {
                "timestamp_s": e.timestamp_s,
                "scenario_id": e.scenario_id,
                "step": e.step,
                "ego_id": e.ego_id,
                "mode": e.mode,
                "speed": e.speed,
                "nav": e.nav,
                "source": e.source,
            }
            for e in items
        ],
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await STATE.attach_ws(ws)
    try:
        while True:
            # We don't expect inbound messages from the client; keep the
            # connection alive and bail on disconnect.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        STATE.detach_ws(ws)


@app.post("/api/negotiate")
async def negotiate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Open a negotiation and block until the operator submits decisions.

    Called by the CoLMDriver agent's HitlDecisionProvider once per group
    that needs human input. Body:

        {
          "scenario_id": "town06_vehicle_1_0",
          "step": 1234,
          "group": [0, 1, 2],
          "comm_info_list": [
            {"ego_id": 0, "ego_speed": 7.4, "ego_intention": "...",
             "other_direction": {...}, "other_distance": {...},
             "other_speed": {...}, "other_intention": {...}},
            ...
          ],
          "timeout_s": 30.0
        }

    Response:

        {"decisions": {"0": {"speed": "KEEP", "nav": "follow the lane"},
                       "1": {"speed": "FASTER", "nav": "..."}, ...},
         "source": "human" | "timeout-fallback"}
    """
    scenario_id = str(payload.get("scenario_id") or "")
    if not scenario_id:
        raise HTTPException(400, "scenario_id is required")
    try:
        step = int(payload["step"])
        group = [int(x) for x in payload["group"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"step and group are required: {exc}")
    comm_info_list = payload.get("comm_info_list") or [{"ego_id": g} for g in group]
    timeout_s = float(payload.get("timeout_s", 30.0))

    # Update top-level scenario state so the UI shows what we're deciding for.
    await STATE.set_step(scenario_id, step)

    decisions = await STATE.open_negotiation(
        scenario_id=scenario_id,
        step=step,
        group=group,
        comm_info_list=list(comm_info_list),
        timeout_s=timeout_s,
    )
    # Find which source we used (the open_negotiation call already logged).
    source = "human"
    if STATE.decision_log:
        # Approximate: the last log entry for this step matches.
        last = STATE.decision_log[-1]
        if last.step == step and last.scenario_id == scenario_id:
            source = last.source
    return {
        "decisions": {str(k): v for k, v in decisions.items()},
        "source": source,
    }


# Convenience alias kept around to exercise the UI without an agent. Same
# implementation as /api/negotiate; useful for `curl` smoke tests.
@app.post("/api/_debug/open_negotiation")
async def _debug_open(payload: Dict[str, Any]) -> Dict[str, Any]:
    scenario_id = str(payload.get("scenario_id", "debug"))
    step = int(payload.get("step", 0))
    group = [int(x) for x in (payload.get("group") or [0, 1])]
    timeout_s = float(payload.get("timeout_s", 30.0))
    await STATE.set_step(scenario_id, step)
    decisions = await STATE.open_negotiation(
        scenario_id=scenario_id,
        step=step,
        group=group,
        comm_info_list=[{"ego_id": gid} for gid in group],
        timeout_s=timeout_s,
    )
    return {"resolved": True, "decisions": decisions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hitl_dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true", help="dev autoreload (uvicorn)")
    p.add_argument(
        "--decision-log",
        default=os.environ.get("HITL_DECISION_LOG"),
        help=(
            "Append-only JSONL path for human decisions. Defaults to "
            "<results-dir>/decisions.jsonl when --results-dir is set; "
            "otherwise $HITL_DECISION_LOG (unset = no file logging)."
        ),
    )
    p.add_argument(
        "--results-dir",
        default=os.environ.get("HITL_RESULTS_DIR"),
        help=(
            "Directory to write the run's results bundle: "
            "decisions.jsonl, telemetry.jsonl, terminal_events.jsonl, "
            "results.json (live snapshot). Defaults to $HITL_RESULTS_DIR."
        ),
    )
    return p.parse_args()


def main() -> int:
    import uvicorn

    args = _parse_args()
    if args.decision_log:
        STATE.decision_log_path = Path(args.decision_log).expanduser().resolve()
        print(f"[hitl_dashboard] decision log: {STATE.decision_log_path}")
    if args.results_dir:
        STATE.set_results_dir(Path(args.results_dir))
        print(f"[hitl_dashboard] results bundle: {STATE.results_dir}")
        print(f"[hitl_dashboard]   telemetry: {STATE.telemetry_log_path}")
        print(f"[hitl_dashboard]   terminal:  {STATE.terminal_log_path}")
        print(f"[hitl_dashboard]   results:   {STATE.results_json_path}")
        print(f"[hitl_dashboard]   decisions: {STATE.decision_log_path}")
    print(
        f"[hitl_dashboard] serving at http://{args.host}:{args.port}  "
        f"(forward via: ssh -L {args.port}:localhost:{args.port} <eval-server>)"
    )
    uvicorn.run(
        "tools.hitl_dashboard.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
