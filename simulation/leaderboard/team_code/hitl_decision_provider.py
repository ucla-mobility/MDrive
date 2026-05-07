"""HITL decision provider — dashboard HTTP client for CoLMDriver.

Used **only** when `COLMDRIVER_HITL=1` is set. When unset, the agent never
imports or instantiates this class, so default CoLMDriver behaviour is
unchanged.

Public surface:

    provider = HitlDecisionProvider.from_env()  # raises if HITL not enabled
    decisions = provider.request_group_decision(
        scenario_id="...",
        step=1234,
        timestamp=61.7,
        group=[0, 1, 2],
        comm_info_list=[...],   # passthrough for the UI
        timeout_s=30.0,
    )
    # -> {0: {"speed": "KEEP", "nav": "follow the lane"}, 1: {...}, 2: {...}}

The call blocks the CARLA tick (matching CoLMDriver's existing synchronous
tick model) until the operator submits or the dashboard's own timeout
fires. On any HTTP failure the provider falls back to a `KEEP` /
`follow the lane` dict for every ego in the group and logs a warning;
this never raises into the agent.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


_DEFAULT_URL = "http://127.0.0.1:8765"
_FALLBACK_SPEED = "KEEP"
_FALLBACK_NAV = "follow the lane"


class HitlDecisionProvider:
    """Minimal blocking HTTP client for `POST /api/negotiate`.

    Uses urllib so the provider works inside the colmdrivermarco2 env
    without adding a dependency on `requests`.
    """

    def __init__(self, *, base_url: str, request_timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        # Note: this is the urllib socket timeout (per-request), NOT the
        # negotiation deadline. The negotiation deadline is server-side and
        # is sent in the request body as `timeout_s`. We let urllib wait a
        # bit longer than the server so the server's own timeout fires
        # first and gives us a clean fallback dict.
        self.request_timeout_s = request_timeout_s

    @classmethod
    def from_env(cls) -> "HitlDecisionProvider":
        """Construct from env: `COLMDRIVER_HITL_URL` (default 127.0.0.1:8765).

        Raises RuntimeError if `COLMDRIVER_HITL` is not set to "1" — callers
        should gate construction on the same env var.
        """
        if os.environ.get("COLMDRIVER_HITL") != "1":
            raise RuntimeError(
                "HitlDecisionProvider must only be used when "
                "COLMDRIVER_HITL=1 is set."
            )
        url = os.environ.get("COLMDRIVER_HITL_URL", _DEFAULT_URL)
        # Keep the urllib timeout generous enough to comfortably outlast the
        # server's negotiation deadline (default 30s) plus jitter.
        timeout = float(os.environ.get("COLMDRIVER_HITL_REQUEST_TIMEOUT_S", "600"))
        return cls(base_url=url, request_timeout_s=timeout)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def request_group_decision(
        self,
        *,
        scenario_id: str,
        step: int,
        timestamp: float,
        group: List[int],
        comm_info_list: List[Dict[str, Any]],
        timeout_s: float = 30.0,
    ) -> Dict[int, Dict[str, str]]:
        """Block until the operator submits decisions, or fall back.

        Returns `{ego_id (int): {"speed": str, "nav": str}}` covering every
        id in `group`. Any ego missing from the server response is filled
        in with the safe fallback so callers never see a KeyError.
        """
        body = {
            "scenario_id": str(scenario_id),
            "step": int(step),
            "timestamp": float(timestamp),
            "group": [int(g) for g in group],
            "comm_info_list": _jsonable(comm_info_list),
            "timeout_s": float(timeout_s),
        }
        url = f"{self.base_url}/api/negotiate"
        try:
            payload = self._post_json(url, body)
        except (urllib_error.URLError, OSError, json.JSONDecodeError) as exc:
            print(
                f"[hitl] WARN dashboard call failed ({exc!r}); using KEEP fallback "
                f"for group={group}",
                file=sys.stderr,
            )
            return self._fallback(group)

        decisions_raw = payload.get("decisions") or {}
        out: Dict[int, Dict[str, str]] = {}
        for g in group:
            entry = decisions_raw.get(str(g)) or decisions_raw.get(g)
            if not isinstance(entry, dict):
                print(
                    f"[hitl] WARN missing decision for ego {g}; falling back.",
                    file=sys.stderr,
                )
                out[g] = {"speed": _FALLBACK_SPEED, "nav": _FALLBACK_NAV}
                continue
            speed = str(entry.get("speed", _FALLBACK_SPEED))
            nav = str(entry.get("nav", _FALLBACK_NAV))
            out[g] = {"speed": speed, "nav": nav}
        source = str(payload.get("source", "human"))
        print(f"[hitl] decisions for group={group}: {out} (source={source})")
        return out

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _post_json(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            method="POST",
            headers={"content-type": "application/json"},
        )
        with urllib_request.urlopen(req, timeout=self.request_timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8") or "{}")

    def _fallback(self, group: List[int]) -> Dict[int, Dict[str, str]]:
        return {
            int(g): {"speed": _FALLBACK_SPEED, "nav": _FALLBACK_NAV}
            for g in group
        }


def _jsonable(obj: Any) -> Any:
    """Coerce numpy / non-stdlib types to JSON-serializable equivalents.

    The agent occasionally builds comm_info_list entries with numpy floats /
    ints. Forcing them through json once on the way out avoids surprises.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        # numpy scalar
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
