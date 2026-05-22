#!/usr/bin/env python3
"""
build_verified_lane_directions.py — build a motion-validated map of lane
travel directions for use by ``fix_actor_yaw_reversal.py``.

CARLA's offline xodr loader (``carla.Map(name, xodr_string)``) returns the
nominal lane travel direction at any point via ``get_waypoint().rotation.yaw``.
This is the right direction for ~91 % of lanes in the v2xpnp scenarioset,
but a handful of map-authoring artifacts in ``ucla_v2.xodr`` cause specific
lanes to report the *opposite* of actual traffic flow.

We cross-validate by walking every moving-vehicle motion segment across the
whole scenarioset, looking up the CARLA waypoint at that midpoint, and
comparing CARLA's stored ``rotation.yaw`` to the observed motion direction.
Per-lane (keyed by ``(road_id, lane_id)``) we tally:

  * ``TRUST_CARLA``    — ≥ 80 % of ≥10 samples agree with CARLA → use as-is.
  * ``INVERT_CARLA``   — ≥ 80 % of ≥10 samples are 180° off CARLA → invert.
  * ``UNRELIABLE``     — mixed signal, do not use lane direction here.
  * ``UNDER_SAMPLED``  — fewer than 10 samples; trust CARLA as default
                         (the global agreement rate is 91 % which is the
                         prior, but flag for review).

Outputs a JSON file (default ``v2xpnp/map/verified_lane_directions.json``).
This file is the cache; rebuild whenever the scenarioset or the xodr
changes.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import glob
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


# Detection thresholds
MIN_STEP_M           = 0.3
WP_MAX_LATERAL_M     = 3.0
AGREE_TOL_DEG        = 30.0
OPP_TOL_DEG          = 30.0
MIN_SAMPLES_PER_LANE = 10
VERIFY_RATIO         = 0.80
DEFAULT_XODR    = Path("v2xpnp/map/ucla_v2.xodr")
DEFAULT_SCENESET = Path("scenarioset/v2xpnp")
DEFAULT_OUT     = Path("v2xpnp/map/verified_lane_directions.json")
DEFAULT_EGG = Path("carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg")


# ---------------------------------------------------------------------------

def _wrap180(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


@contextlib.contextmanager
def _silenced_stderr():
    """Redirect C-level stderr (fd 2) so the CARLA xodr loader's
    'Traffic sign overlaps a driving lane' warnings don't drown the log.
    """
    old = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old, 2)
        os.close(devnull)
        os.close(old)


def _load_carla_map(xodr_path: Path):
    """Load the OpenDRIVE map offline via the CARLA Python API."""
    # __file__ lives at v2xpnp/scripts/<this>.py — walk up three levels
    # to reach the repo root before joining the relative egg path.
    egg = (Path(__file__).resolve().parent.parent.parent / DEFAULT_EGG).resolve()
    if not egg.exists():
        # Fall back to whatever CARLA pip-install we have
        try:
            import carla  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                f"CARLA egg not found at {egg} and no system carla module is "
                f"importable. Cannot load xodr offline. ({exc})"
            )
    else:
        sys.path.insert(0, str(egg))
    import carla
    xodr_str = xodr_path.read_text()
    with _silenced_stderr():
        m = carla.Map(xodr_path.stem, xodr_str)
    return carla, m


def _iter_motion_segments(sceneset: Path):
    """Yield ``(road_dir_lookup_xy, motion_yaw_deg)`` per motion step ≥ MIN_STEP_M
    of every non-walker actor in every scene. Caller provides the CARLA map
    lookup; this generator only yields raw data so the caller can choose to
    batch waypoint queries.
    """
    for scene_dir in sorted(sceneset.iterdir()):
        if not scene_dir.is_dir(): continue
        if scene_dir.name == "carla_routes_patched": continue
        for xml_path in glob.glob(f"{scene_dir}/**/*.xml", recursive=True):
            if xml_path.endswith(".preflip") or xml_path.endswith("_REPLAY.xml"):
                continue
            if "/walker/" in xml_path or "/cyclist/" in xml_path:
                continue
            try:
                tree = ET.parse(xml_path)
            except Exception:
                continue
            root = tree.getroot()
            route = root.find("route") if root.tag != "route" else root
            if route is None:
                continue
            if (route.attrib.get("role") or "").lower() in ("walker", "cyclist"):
                continue
            wps = route.findall("waypoint")
            for i in range(len(wps) - 1):
                try:
                    xa = float(wps[i].attrib["x"])
                    ya = float(wps[i].attrib["y"])
                    xb = float(wps[i + 1].attrib["x"])
                    yb = float(wps[i + 1].attrib["y"])
                except (KeyError, ValueError):
                    continue
                dx, dy = xb - xa, yb - ya
                step = math.hypot(dx, dy)
                if step < MIN_STEP_M:
                    continue
                yield (
                    (xa + xb) / 2.0,
                    (ya + yb) / 2.0,
                    math.degrees(math.atan2(dy, dx)),
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xodr", type=Path, default=DEFAULT_XODR)
    ap.add_argument("--scenarioset", type=Path, default=DEFAULT_SCENESET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_LANE,
                    help=f"Per-lane minimum observed samples (default: "
                         f"{MIN_SAMPLES_PER_LANE})")
    ap.add_argument("--verify-ratio", type=float, default=VERIFY_RATIO,
                    help=f"Minimum agree- or oppose-ratio to label a lane "
                         f"(default: {VERIFY_RATIO})")
    args = ap.parse_args()

    if not args.xodr.is_file():
        print(f"ERROR: xodr not found at {args.xodr}", file=sys.stderr)
        sys.exit(2)
    if not args.scenarioset.is_dir():
        print(f"ERROR: scenarioset not found at {args.scenarioset}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading {args.xodr}…")
    carla, m = _load_carla_map(args.xodr)
    print(f"  loaded map: {m.name}")

    print(f"Scanning {args.scenarioset}…")
    per_lane: Dict[Tuple[int, int], Dict] = collections.defaultdict(
        lambda: {"n": 0, "agree": 0, "oppose": 0, "other": 0,
                 "agree_yaw_sin": 0.0, "agree_yaw_cos": 0.0,
                 "oppose_yaw_sin": 0.0, "oppose_yaw_cos": 0.0}
    )
    n_total = n_no_lane = 0
    n_seen = 0
    for mx, my, motion_yaw in _iter_motion_segments(args.scenarioset):
        n_seen += 1
        if n_seen % 5000 == 0:
            print(f"  [scan] processed {n_seen} segments…", flush=True)
        wp = m.get_waypoint(carla.Location(x=mx, y=my, z=0.0),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving)
        if wp is None:
            n_no_lane += 1; continue
        wpx, wpy = wp.transform.location.x, wp.transform.location.y
        if math.hypot(wpx - mx, wpy - my) > WP_MAX_LATERAL_M:
            n_no_lane += 1; continue
        wp_yaw = wp.transform.rotation.yaw
        diff = abs(_wrap180(motion_yaw - wp_yaw))
        key = (wp.road_id, wp.lane_id)
        entry = per_lane[key]
        entry["n"] += 1
        if diff < AGREE_TOL_DEG:
            entry["agree"] += 1
            entry["agree_yaw_sin"] += math.sin(math.radians(motion_yaw))
            entry["agree_yaw_cos"] += math.cos(math.radians(motion_yaw))
        elif abs(diff - 180.0) < OPP_TOL_DEG:
            entry["oppose"] += 1
            entry["oppose_yaw_sin"] += math.sin(math.radians(motion_yaw))
            entry["oppose_yaw_cos"] += math.cos(math.radians(motion_yaw))
        else:
            entry["other"] += 1
        n_total += 1

    # ── Verdict per lane ────────────────────────────────────────────────
    out_lanes: Dict[str, Dict] = {}
    n_trust = n_invert = n_unreliable = n_under = 0
    for (rid, lid), e in per_lane.items():
        n = e["n"]
        if n < args.min_samples:
            verdict = "UNDER_SAMPLED"
            n_under += 1
        elif e["agree"] / n >= args.verify_ratio:
            verdict = "TRUST_CARLA"
            n_trust += 1
        elif e["oppose"] / n >= args.verify_ratio:
            verdict = "INVERT_CARLA"
            n_invert += 1
        else:
            verdict = "UNRELIABLE"
            n_unreliable += 1
        # Aggregate motion direction (only the dominant cluster's mean)
        if verdict == "TRUST_CARLA":
            ay = math.degrees(math.atan2(e["agree_yaw_sin"], e["agree_yaw_cos"]))
        elif verdict == "INVERT_CARLA":
            ay = math.degrees(math.atan2(e["oppose_yaw_sin"], e["oppose_yaw_cos"]))
        else:
            ay = None
        out_lanes[f"{rid}:{lid}"] = {
            "road_id": rid,
            "lane_id": lid,
            "n_samples": n,
            "n_agree": e["agree"],
            "n_oppose": e["oppose"],
            "n_other": e["other"],
            "verdict": verdict,
            "motion_forward_yaw_deg": ay,
        }

    payload = {
        "xodr": str(args.xodr.resolve()),
        "scenarioset": str(args.scenarioset.resolve()),
        "thresholds": {
            "min_samples": args.min_samples,
            "verify_ratio": args.verify_ratio,
            "agree_tol_deg": AGREE_TOL_DEG,
            "opp_tol_deg": OPP_TOL_DEG,
            "min_step_m": MIN_STEP_M,
            "wp_max_lateral_m": WP_MAX_LATERAL_M,
        },
        "totals": {
            "segments_scanned": n_seen,
            "segments_matched": n_total,
            "segments_no_lane": n_no_lane,
            "lanes_total": len(per_lane),
            "lanes_trust": n_trust,
            "lanes_invert": n_invert,
            "lanes_unreliable": n_unreliable,
            "lanes_under_sampled": n_under,
        },
        "lanes": out_lanes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Wrote {args.out}")
    print(f"  scanned segments      : {n_seen}")
    print(f"  matched to a lane     : {n_total}  ({100*n_total/max(1,n_seen):.1f}%)")
    print(f"  no driving lane within {WP_MAX_LATERAL_M}m : {n_no_lane}")
    print(f"  lanes total           : {len(per_lane)}")
    print(f"  TRUST_CARLA           : {n_trust}")
    print(f"  INVERT_CARLA          : {n_invert}")
    print(f"  UNRELIABLE            : {n_unreliable}")
    print(f"  UNDER_SAMPLED         : {n_under}")
    if n_invert:
        print(f"\nLanes flagged for inversion (CARLA stored direction opposite of motion):")
        for k, v in out_lanes.items():
            if v["verdict"] == "INVERT_CARLA":
                print(f"  road {v['road_id']:>4d} lane {v['lane_id']:>3d}  "
                      f"({v['n_samples']:>3d} samples; {v['n_oppose']}/{v['n_samples']} opposite)")
    if n_unreliable:
        print(f"\nLanes flagged UNRELIABLE (mixed motion — direction can't be decided):")
        for k, v in sorted(out_lanes.items(), key=lambda kv: -kv[1]["n_samples"]):
            if v["verdict"] == "UNRELIABLE":
                print(f"  road {v['road_id']:>4d} lane {v['lane_id']:>3d}  "
                      f"({v['n_samples']:>3d} samples; agree={v['n_agree']} "
                      f"oppose={v['n_oppose']} other={v['n_other']})")


if __name__ == "__main__":
    main()
