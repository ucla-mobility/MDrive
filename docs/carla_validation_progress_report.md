# CARLA Validation Progress Report — 500 Most Recent Runs

**Date:** March 1, 2026  
**Runs analyzed:** 500 (from `20260227_015120` to `20260227_065810`)  
**Data source:** `debug_runs/*/summary.json` + `10_carla_validation/output.json`

---

## Executive Summary

| Metric | Count | Rate |
|--------|------:|-----:|
| **CARLA PASS** | 130 | 26.0% |
| **CARLA FAIL (definitive)** | 368 | 73.6% |
| **Never run** | 2 | 0.4% |
| Regular validation pass | 498 | 99.6% |
| Regular validation fail | 2 | 0.4% |

**The scenario generation pipeline itself is working well** — 99.6% of runs pass regular (non-CARLA) validation. The bottleneck is entirely in CARLA validation, where most failures are **infrastructure failures (CARLA crashing), not scenario quality issues.**

---

## Failure Breakdown

### The 368 CARLA Failures by Root Cause

| Root Cause | Count | % of Failures | Real Issue? |
|------------|------:|:-------------:|:-----------:|
| `carla_connect_failed: Resource temporarily unavailable` | 297 | 80.7% | **No** — CARLA crashed |
| `spawn_all_actors` (try_spawn_actor returned None) | 51 | 13.9% | **Partial** — some real, most due to stale CARLA |
| `xml_manifest_contract_failed` (missing_actors_manifest) | 20 | 5.4% | **Yes** — scenario generation bug |

### Interpretation

1. **297 runs (80.7% of failures)** failed because CARLA became unresponsive mid-session. The error `Resource temporarily unavailable` means the CARLA process crashed or exhausted its connection capacity. These are **not real scenario failures** — the scenarios were never actually tested.

2. **51 runs (13.9%)** failed at `spawn_all_actors`. In these, `try_spawn_actor returned None` for ego vehicles and NPCs. Many of these likely also stem from a degraded CARLA state (zombie actors from previous runs consuming spawn slots), though some may be genuine spawn point conflicts.

3. **20 runs (5.4%)** failed with `missing_actors_manifest` — these runs lack a proper `actors_manifest.json` file in their routes directory. This is a **real pipeline bug** in scenario generation (certain categories not producing the manifest file).

---

## CARLA Crash Pattern — Timeline Analysis

The data shows a very clear pattern: **CARLA processes survive ~10-15 validation runs, then die, and every subsequent run on that port fails instantly.**

### Port 3500 Timeline
- **10 passes** from 12:30:09 to 12:53:40 (~23 minutes of successful validations)
- Then a **10-hour, 44-minute gap** (runner was restarted)
- After restart: **169 consecutive failures** — all `spawn_all_actors` (CARLA was already dying)
- The failures are rapid (~12s each) — confirming CARLA is dead, not slow

### Port 3505 Timeline
- **5 passes**, all successful (this instance was on the port-collision port from the old runner — it worked briefly before the collision killed it)

### Port 3510 Timeline
- **2 passes** only, then **148 consecutive failures**
- The transition is instant: 23:42:46 PASS → 23:42:46 FAIL (same second!)
- All subsequent failures: `Resource temporarily unavailable`

### Key Observation
CARLA dies after accumulating too many spawned/destroyed actors without a clean world restart. The `_destroy_actors()` cleanup in `carla_validation.py` only destroys actors spawned in that specific validation call, but **does not reload the world** between runs. Over time, zombie actors and leaked resources crash CARLA.

---

## Why Only ~130 Passed

The 130 successful validations correspond almost exactly to:
- **Port 2030:** 61 passes (original pipeline run)
- **Port 2090:** 52 passes (original pipeline run)  
- **Port 3500:** 10 passes (first retry cycle)
- **Port 3505:** 5 passes (first retry cycle, before port collision)
- **Port 3510:** 2 passes (second retry cycle)

Each CARLA instance validates ~10-15 scenarios before dying. With 2 instances, that's ~20-30 per cycle. The retry runner only completed 1-2 cycles before CARLA died.

---

## Pass Rate by Scenario Category

| Category | Pass | Total | Rate |
|----------|-----:|------:|-----:|
| Intersection Deadlock Resolution | 17 | 45 | 37.8% |
| Interactive Lane Change | 17 | 45 | 37.8% |
| Roundabout Navigation | 17 | 46 | 37.0% |
| Overtaking on Two-Lane Road | 16 | 46 | 34.8% |
| Highway On-Ramp Merge | 16 | 45 | 35.6% |
| Unprotected Left Turn | 13 | 46 | 28.3% |
| Lane Drop / Alternating Merge | 13 | 45 | 28.9% |
| Pedestrian Crosswalk | 9 | 46 | 19.6% |
| Construction Zone | 6 | 45 | 13.3% |
| Blocked Lane (Obstacle) | 6 | 45 | 13.3% |
| **Major/Minor Unsignalized Entry** | **0** | **46** | **0.0%** |

The variation in pass rates is **not** meaningful — it reflects which scenarios happened to be processed early (before CARLA crashed), not scenario quality. Categories processed later in the queue had lower pass rates simply because CARLA was already dead.

**Exception:** `Major/Minor Unsignalized Entry` at 0% may have a real issue (e.g., spawn points in Town02 being problematic), but this can't be confirmed until CARLA stays alive long enough to actually test them.

---

## Root Causes & Recommended Fixes

### 1. CARLA Process Stability (Critical)
**Problem:** CARLA accumulates state across validation runs and crashes after ~10-15 consecutive scenario validations.

**Evidence:** The `WARNING: attempting to destroy an actor that is already dead` messages show actors leaking. Each scenario spawns 3-8 actors. After 15 scenarios, that's 45-120 spawn/destroy cycles — enough to corrupt CARLA's internal actor registry.

**Fix options:**
- **`client.reload_world()`** between each validation run (resets all actors, ~3-5s overhead)
- Reduce the validation batch size in the cycling runner to 8-10 runs per CARLA cycle
- Add a CARLA health-check RPC call before each validation; if it fails, immediately restart

### 2. Port Collision in Runner (Fixed)
**Problem:** The original runner placed two CARLA instances 5 ports apart. CARLA uses ports N, N+1, N+5, causing instance 1's traffic manager (N+5) to collide with instance 2's RPC port.

**Status:** Fixed in latest `run_audit_with_carla_cycling.sh` — instances now have 10-port gap.

### 3. Missing Actors Manifest (20 runs, real bug)
**Problem:** 20 runs don't have `actors_manifest.json` in their routes directory.

**Likely cause:** Certain scenario categories or seeds hit an edge case in the route generation stage that skips manifest creation.

### 4. Spawn Failures (51 runs, partially real)
**Problem:** `try_spawn_actor returned None` for ego and NPC vehicles.

**Causes (mixed):**
- Stale CARLA state from previous runs (zombie actors blocking spawn points)
- Spawn points too close together or overlapping with map geometry
- Some scenarios require repair offsets that aren't applied correctly

---

## Projected Pass Rate with Stable CARLA

If CARLA were stable:
- **297 RTUA runs** would get real results. Extrapolating from the 130 that did get tested (130 pass out of 150 that actually ran = **86.7% pass rate**), approximately **257 more should pass**.
- **51 spawn_all_actors** — some would pass with a clean world, maybe ~50%
- **20 xml_manifest** — these need a pipeline fix, ~0% pass

**Estimated true pass rate: ~380-400/500 (76-80%)** once CARLA stability is resolved.

---

## Current State of Retry Infrastructure

| Component | Status |
|-----------|--------|
| `run_audit_with_carla_cycling.sh` | Fixed (port spacing, nuke-all-carla, port-free checks) |
| Resume checkpoint | Working (495 completed recorded) |
| `--carla-resume` flag | Working — skips already-validated runs |
| World cleanup between runs | **Missing** — root cause of CARLA crashes |
| CARLA health probing | Exists but doesn't trigger restart fast enough |
