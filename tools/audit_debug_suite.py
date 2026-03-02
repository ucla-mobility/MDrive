#!/usr/bin/env python3
"""
Audit Debug Suite - Comprehensive visualization and analysis for scenario generation pipeline.

Generates an interactive HTML dashboard to analyze:
- Per-scenario success/failure breakdown
- Pipeline stage-by-stage analysis
- Validation metrics and issue patterns
- Visual artifacts (scene PNGs, path visualizations)
- Aggregate statistics and weak point identification
- Repair loop effectiveness
- Timing analysis

Usage:
    python tools/audit_debug_suite.py <run_directory>
    python tools/audit_debug_suite.py benchmark_runs/audit_20260121_120617
    python tools/audit_debug_suite.py benchmark_runs/audit_20260121_120617 --open
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    name: str
    success: bool
    started: bool = True
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_s: Optional[float] = None


@dataclass
class ValidationIssue:
    """A validation issue from scene validator."""
    severity: str
    category: str
    message: str
    expected: str = ""
    actual: str = ""
    suggestion: str = ""


@dataclass
class ScenarioAnalysis:
    """Complete analysis of a single scenario run."""
    run_key: str
    category: str
    variant_index: int
    success: bool
    validation_score: Optional[float]
    failure_state: Optional[str]
    failure_reason: Optional[str]
    
    # Pipeline stages
    stages: List[StageResult] = field(default_factory=list)
    
    # Attempts
    total_attempts: int = 0
    successful_attempt: Optional[int] = None
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Repair metrics
    repair_outer_attempts: int = 0
    repair_schema_attempts: int = 0
    repair_object_stage1_json: int = 0
    repair_object_stage1_evidence: int = 0
    repair_object_stage2_json: int = 0
    repair_object_stage2_validation: int = 0
    template_fallback: bool = False
    
    # Validation
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    
    # Artifacts
    scene_png_path: Optional[str] = None
    scene_json_path: Optional[str] = None
    spec_json_path: Optional[str] = None
    routes_dir: Optional[str] = None
    run_dir: Optional[str] = None
    
    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    elapsed_s: Optional[float] = None
    
    # Baseline validation
    baseline_status: Optional[str] = None
    baseline_rc: Optional[float] = None
    baseline_ds: Optional[float] = None
    baseline_near_miss: Optional[bool] = None
    baseline_min_ttc: Optional[float] = None
    
    # Best-of ranking
    best_rank: Optional[int] = None
    kept_best: bool = False
    output_hash: Optional[str] = None


@dataclass
class CategorySummary:
    """Summary statistics for a category."""
    name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    avg_validation_score: float = 0.0
    avg_attempts: float = 0.0
    avg_duration_s: float = 0.0
    failure_stage_counts: Dict[str, int] = field(default_factory=dict)
    kept_best_count: int = 0
    
    # Repair effectiveness
    total_repairs: int = 0
    repairs_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Common issues
    common_issues: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class RunAnalysis:
    """Complete analysis of an audit run."""
    run_id: str
    run_dir: Path
    created_at: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    scenarios: List[ScenarioAnalysis] = field(default_factory=list)
    category_summaries: Dict[str, CategorySummary] = field(default_factory=dict)
    
    # Aggregate metrics
    total_scenarios: int = 0
    successful_scenarios: int = 0
    failed_scenarios: int = 0
    overall_success_rate: float = 0.0
    avg_validation_score: float = 0.0
    total_duration_s: float = 0.0
    
    # Pipeline health
    stage_success_rates: Dict[str, float] = field(default_factory=dict)
    common_failure_stages: List[Tuple[str, int]] = field(default_factory=list)
    common_failure_reasons: List[Tuple[str, int]] = field(default_factory=list)
    
    # Validation patterns
    common_validation_issues: List[Tuple[str, int]] = field(default_factory=list)
    issue_severity_counts: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# Data Loading
# =============================================================================

def load_run_analysis(run_dir: Path) -> RunAnalysis:
    """Load and analyze a complete audit run."""
    run_id = run_dir.name
    analysis = RunAnalysis(run_id=run_id, run_dir=run_dir)
    
    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            analysis.config = json.load(f)
            analysis.created_at = analysis.config.get("created_at")
    
    # Load CSV data
    csv_path = run_dir / "audit.csv"
    if csv_path.exists():
        scenarios = _load_scenarios_from_csv(csv_path, run_dir)
        analysis.scenarios = scenarios
    
    # Also scan run directories for additional data
    runs_dir = run_dir / "runs"
    if runs_dir.exists():
        _enrich_scenarios_from_dirs(analysis.scenarios, runs_dir)
    
    # Compute aggregate metrics
    _compute_aggregates(analysis)
    
    return analysis


def _load_scenarios_from_csv(csv_path: Path, run_dir: Path) -> List[ScenarioAnalysis]:
    """Load scenario data from CSV."""
    scenarios = []
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") != "scenario_run":
                continue
            
            scenario = ScenarioAnalysis(
                run_key=row.get("run_key", ""),
                category=row.get("category", ""),
                variant_index=int(row.get("variant_index", 0) or 0),
                success=row.get("scenario_generation_success", "").lower() == "true",
                validation_score=_parse_float(row.get("validation_score")),
                failure_state=row.get("failure_state") or None,
                failure_reason=row.get("failure_reason") or None,
            )
            
            # Repair metrics
            scenario.repair_outer_attempts = int(row.get("repair_outer_attempts", 0) or 0)
            scenario.repair_schema_attempts = int(row.get("repair_schema_attempts", 0) or 0)
            scenario.repair_object_stage1_json = int(row.get("repair_object_stage1_json", 0) or 0)
            scenario.repair_object_stage1_evidence = int(row.get("repair_object_stage1_evidence", 0) or 0)
            scenario.repair_object_stage2_json = int(row.get("repair_object_stage2_json", 0) or 0)
            scenario.repair_object_stage2_validation = int(row.get("repair_object_stage2_validation", 0) or 0)
            scenario.template_fallback = int(row.get("schema_template_fallback", 0) or 0) > 0
            scenario.total_attempts = int(row.get("pipeline_attempts_used", 0) or 0)
            
            # Paths
            scenario.scene_png_path = row.get("scene_png_path") or None
            scenario.scene_json_path = row.get("scene_objects_path") or None
            scenario.spec_json_path = row.get("scenario_spec_path") or None
            scenario.routes_dir = row.get("routes_dir") or None
            
            # Timing
            scenario.start_time = row.get("start_time") or None
            scenario.end_time = row.get("end_time") or None
            scenario.elapsed_s = _parse_float(row.get("elapsed_s"))
            
            # Best-of
            scenario.best_rank = _parse_int(row.get("best_rank"))
            scenario.kept_best = row.get("kept_best", "").lower() == "true"
            
            # Set run_dir
            scenario.run_dir = str(run_dir / "runs" / scenario.run_key)
            
            scenarios.append(scenario)
    
    return scenarios


def _enrich_scenarios_from_dirs(scenarios: List[ScenarioAnalysis], runs_dir: Path) -> None:
    """Enrich scenario data from run directories."""
    scenario_by_key = {s.run_key: s for s in scenarios}
    
    for run_key_dir in runs_dir.iterdir():
        if not run_key_dir.is_dir():
            continue
        run_key = run_key_dir.name
        scenario = scenario_by_key.get(run_key)
        if not scenario:
            continue
        
        scenario.run_dir = str(run_key_dir)
        
        # Load status.json for detailed info
        status_path = run_key_dir / "status.json"
        if status_path.exists():
            _enrich_from_status(scenario, status_path)
        
        # Load validation report if exists
        _load_validation_issues(scenario, run_key_dir)
        
        # Detect pipeline stage artifacts
        _detect_stage_artifacts(scenario, run_key_dir)
        
        # Load baseline metrics
        _load_baseline_metrics(scenario, run_key_dir)


def _enrich_from_status(scenario: ScenarioAnalysis, status_path: Path) -> None:
    """Enrich scenario from status.json."""
    try:
        with open(status_path) as f:
            status = json.load(f)
    except Exception:
        return
    
    scenario.attempt_history = status.get("attempt_history", [])
    
    # Build stage results from state history
    completed = set(status.get("completed_states", []))
    failure_state = status.get("failure_state")
    
    stages = [
        "scenario_generation",
        "repair_loops",
        "route_generation",
        "baseline_validation",
        "carla_simulation",
        "video_generation",
        "csv_commit",
    ]
    
    for stage_name in stages:
        started = stage_name in completed or stage_name == failure_state
        success = stage_name in completed
        error = None
        if stage_name == failure_state:
            started = True
            success = False
            error = status.get("failure_reason")
        
        scenario.stages.append(StageResult(
            name=stage_name,
            success=success,
            started=started,
            error=error,
        ))
    
    # Extract metrics from status
    metrics = status.get("metrics", {})
    baseline = metrics.get("baseline", {})
    if baseline:
        scenario.baseline_status = baseline.get("status")
        scenario.baseline_rc = _parse_float(baseline.get("rc"))
        scenario.baseline_ds = _parse_float(baseline.get("ds"))
        scenario.baseline_near_miss = baseline.get("near_miss")
        scenario.baseline_min_ttc = _parse_float(baseline.get("min_ttc"))


def _load_validation_issues(scenario: ScenarioAnalysis, run_dir: Path) -> None:
    """Load validation issues from reports."""
    # Check various possible locations
    possible_paths = [
        run_dir / "pipeline" / f"{scenario.run_key}_attempt{scenario.total_attempts}" / "validation_report.json",
    ]
    
    # Also search pipeline directory
    pipeline_dir = run_dir / "pipeline"
    if pipeline_dir.exists():
        for attempt_dir in pipeline_dir.iterdir():
            if attempt_dir.is_dir():
                report_path = attempt_dir / "validation_report.json"
                if report_path.exists():
                    possible_paths.insert(0, report_path)
    
    for report_path in possible_paths:
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                
                for issue in report.get("issues", []):
                    scenario.validation_issues.append(ValidationIssue(
                        severity=issue.get("severity", "unknown"),
                        category=issue.get("category", "unknown"),
                        message=issue.get("message", ""),
                        expected=issue.get("expected", ""),
                        actual=issue.get("actual", ""),
                        suggestion=issue.get("suggestion", ""),
                    ))
                break
            except Exception:
                pass


def _detect_stage_artifacts(scenario: ScenarioAnalysis, run_dir: Path) -> None:
    """Detect which artifacts exist for each stage."""
    pipeline_dir = run_dir / "pipeline"
    if not pipeline_dir.exists():
        return
    
    # Find the latest/successful attempt directory
    attempt_dirs = sorted(pipeline_dir.iterdir(), reverse=True)
    if not attempt_dirs:
        return
    
    latest_dir = None
    for d in attempt_dirs:
        if d.is_dir() and (d / "scene_objects.json").exists():
            latest_dir = d
            break
    
    if latest_dir is None and attempt_dirs:
        latest_dir = attempt_dirs[0] if attempt_dirs[0].is_dir() else None
    
    if latest_dir is None:
        return
    
    # Map artifacts to stages
    artifact_stage_map = {
        "scenario_spec.json": "scenario_generation",
        "legal_paths_prompt.txt": "scenario_generation",
        "legal_paths_detailed.json": "scenario_generation",
        "legal_paths_all.png": "scenario_generation",
        "picked_paths_detailed.json": "scenario_generation",
        "picked_paths_viz.png": "scenario_generation",
        "picked_paths_refined.json": "scenario_generation",
        "picked_paths_refined_viz.png": "scenario_generation",
        "scene_objects.json": "scenario_generation",
        "scene_objects.png": "scenario_generation",
        "validation_report.json": "scenario_generation",
    }
    
    # Also check routes directory
    routes_dir = run_dir / "routes"
    
    stage_artifacts = defaultdict(list)
    
    for artifact, stage in artifact_stage_map.items():
        artifact_path = latest_dir / artifact
        if artifact_path.exists():
            stage_artifacts[stage].append(str(artifact_path))
    
    if routes_dir.exists():
        route_files = list(routes_dir.glob("*.xml"))
        for rf in route_files:
            stage_artifacts["route_generation"].append(str(rf))
    
    # Update stages with artifacts
    for stage in scenario.stages:
        stage.artifacts = stage_artifacts.get(stage.name, [])


def _load_baseline_metrics(scenario: ScenarioAnalysis, run_dir: Path) -> None:
    """Load baseline validation metrics."""
    routes_dir = run_dir / "routes"
    events_path = routes_dir / "baseline_events.json"
    
    if events_path.exists():
        try:
            with open(events_path) as f:
                events = json.load(f)
            
            # Count collision and near-miss events
            for stage in scenario.stages:
                if stage.name == "baseline_validation":
                    stage.metrics["collision_count"] = len(events.get("collisions", []))
                    stage.metrics["near_miss_count"] = len(events.get("near_miss_hits", []))
                    break
        except Exception:
            pass


def _parse_float(val: Any) -> Optional[float]:
    """Parse a float from various formats."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_int(val: Any) -> Optional[int]:
    """Parse an int from various formats."""
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _compute_aggregates(analysis: RunAnalysis) -> None:
    """Compute aggregate metrics for the analysis."""
    if not analysis.scenarios:
        return
    
    analysis.total_scenarios = len(analysis.scenarios)
    analysis.successful_scenarios = sum(1 for s in analysis.scenarios if s.success)
    analysis.failed_scenarios = analysis.total_scenarios - analysis.successful_scenarios
    analysis.overall_success_rate = analysis.successful_scenarios / analysis.total_scenarios if analysis.total_scenarios else 0
    
    # Average validation score
    scores = [s.validation_score for s in analysis.scenarios if s.validation_score is not None]
    analysis.avg_validation_score = sum(scores) / len(scores) if scores else 0
    
    # Total duration
    durations = [s.elapsed_s for s in analysis.scenarios if s.elapsed_s is not None]
    analysis.total_duration_s = sum(durations)
    
    # Stage success rates
    stage_counts = defaultdict(lambda: {"total": 0, "success": 0})
    for scenario in analysis.scenarios:
        for stage in scenario.stages:
            if stage.started:
                stage_counts[stage.name]["total"] += 1
                if stage.success:
                    stage_counts[stage.name]["success"] += 1
    
    for stage_name, counts in stage_counts.items():
        rate = counts["success"] / counts["total"] if counts["total"] else 0
        analysis.stage_success_rates[stage_name] = rate
    
    # Common failure stages
    failure_stage_counter = Counter(
        s.failure_state for s in analysis.scenarios if s.failure_state
    )
    analysis.common_failure_stages = failure_stage_counter.most_common(10)
    
    # Common failure reasons
    failure_reason_counter = Counter(
        s.failure_reason for s in analysis.scenarios if s.failure_reason
    )
    analysis.common_failure_reasons = failure_reason_counter.most_common(10)
    
    # Validation issues
    all_issues = []
    for scenario in analysis.scenarios:
        for issue in scenario.validation_issues:
            all_issues.append(f"{issue.category}: {issue.message[:50]}")
    issue_counter = Counter(all_issues)
    analysis.common_validation_issues = issue_counter.most_common(15)
    
    # Issue severity counts
    severity_counter = Counter(
        issue.severity for s in analysis.scenarios for issue in s.validation_issues
    )
    analysis.issue_severity_counts = dict(severity_counter)
    
    # Category summaries
    category_scenarios = defaultdict(list)
    for scenario in analysis.scenarios:
        category_scenarios[scenario.category].append(scenario)
    
    for cat_name, cat_scenarios in category_scenarios.items():
        summary = CategorySummary(name=cat_name)
        summary.total_runs = len(cat_scenarios)
        summary.successful_runs = sum(1 for s in cat_scenarios if s.success)
        summary.failed_runs = summary.total_runs - summary.successful_runs
        
        scores = [s.validation_score for s in cat_scenarios if s.validation_score is not None]
        summary.avg_validation_score = sum(scores) / len(scores) if scores else 0
        
        attempts = [s.total_attempts for s in cat_scenarios if s.total_attempts]
        summary.avg_attempts = sum(attempts) / len(attempts) if attempts else 0
        
        durations = [s.elapsed_s for s in cat_scenarios if s.elapsed_s is not None]
        summary.avg_duration_s = sum(durations) / len(durations) if durations else 0
        
        # Failure stages
        for s in cat_scenarios:
            if s.failure_state:
                summary.failure_stage_counts[s.failure_state] = summary.failure_stage_counts.get(s.failure_state, 0) + 1
        
        # Kept best
        summary.kept_best_count = sum(1 for s in cat_scenarios if s.kept_best)
        
        # Repairs
        for s in cat_scenarios:
            total_repairs = (
                s.repair_outer_attempts +
                s.repair_schema_attempts +
                s.repair_object_stage1_json +
                s.repair_object_stage1_evidence +
                s.repair_object_stage2_json +
                s.repair_object_stage2_validation
            )
            summary.total_repairs += total_repairs
            summary.repairs_by_type["outer"] = summary.repairs_by_type.get("outer", 0) + s.repair_outer_attempts
            summary.repairs_by_type["schema"] = summary.repairs_by_type.get("schema", 0) + s.repair_schema_attempts
            summary.repairs_by_type["object_stage1_json"] = summary.repairs_by_type.get("object_stage1_json", 0) + s.repair_object_stage1_json
            summary.repairs_by_type["object_stage1_evidence"] = summary.repairs_by_type.get("object_stage1_evidence", 0) + s.repair_object_stage1_evidence
            summary.repairs_by_type["object_stage2_json"] = summary.repairs_by_type.get("object_stage2_json", 0) + s.repair_object_stage2_json
            summary.repairs_by_type["object_stage2_validation"] = summary.repairs_by_type.get("object_stage2_validation", 0) + s.repair_object_stage2_validation
        
        # Common issues
        cat_issues = []
        for s in cat_scenarios:
            for issue in s.validation_issues:
                cat_issues.append(f"{issue.category}: {issue.message[:40]}")
        summary.common_issues = Counter(cat_issues).most_common(5)
        
        analysis.category_summaries[cat_name] = summary


# =============================================================================
# HTML Generation
# =============================================================================

def _embed_image(path: str) -> str:
    """Embed an image as base64 data URI."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = Path(path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""


def _format_duration(seconds: Optional[float]) -> str:
    """Format duration in human-readable form."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.0f}s"


def _get_status_class(success: bool, partial: bool = False) -> str:
    """Get CSS class for status indicators."""
    if success:
        return "success"
    if partial:
        return "warning"
    return "failure"


def _get_score_class(score: Optional[float]) -> str:
    """Get CSS class based on validation score."""
    if score is None:
        return "neutral"
    if score >= 0.8:
        return "success"
    if score >= 0.5:
        return "warning"
    return "failure"


def generate_html_report(analysis: RunAnalysis, output_path: Path) -> None:
    """Generate comprehensive HTML debug report."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audit Debug Suite - {analysis.run_id}</title>
    <style>
{_get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Audit Debug Suite</h1>
            <p class="run-info">Run ID: <code>{analysis.run_id}</code> | Created: {analysis.created_at or 'Unknown'}</p>
        </header>
        
        <nav class="tabs">
            <button class="tab-btn active" data-tab="overview">📊 Overview</button>
            <button class="tab-btn" data-tab="categories">📁 Categories</button>
            <button class="tab-btn" data-tab="scenarios">🎬 Scenarios</button>
            <button class="tab-btn" data-tab="pipeline">🔧 Pipeline Health</button>
            <button class="tab-btn" data-tab="issues">⚠️ Issues</button>
        </nav>
        
        <main>
            <section id="overview" class="tab-content active">
                {_generate_overview_section(analysis)}
            </section>
            
            <section id="categories" class="tab-content">
                {_generate_categories_section(analysis)}
            </section>
            
            <section id="scenarios" class="tab-content">
                {_generate_scenarios_section(analysis)}
            </section>
            
            <section id="pipeline" class="tab-content">
                {_generate_pipeline_section(analysis)}
            </section>
            
            <section id="issues" class="tab-content">
                {_generate_issues_section(analysis)}
            </section>
        </main>
        
        <footer>
            <p>Generated by Audit Debug Suite | {datetime.now().isoformat()}</p>
        </footer>
    </div>
    
    <script>
{_get_js()}
    </script>
</body>
</html>
"""
    
    output_path.write_text(html, encoding="utf-8")


def _get_css() -> str:
    """Get CSS styles for the report."""
    return """
        :root {
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --bg-hover: #1f3460;
            --text-primary: #e8e8e8;
            --text-secondary: #a0a0a0;
            --accent: #4a90d9;
            --success: #4caf50;
            --warning: #ff9800;
            --failure: #f44336;
            --neutral: #9e9e9e;
            --border: #2d3a5a;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
        }
        
        header h1 { color: var(--accent); font-size: 2.5em; }
        
        .run-info {
            color: var(--text-secondary);
            margin-top: 10px;
        }
        
        code {
            background: var(--bg-hover);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Fira Code', 'Consolas', monospace;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .tab-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.2s;
        }
        
        .tab-btn:hover { background: var(--bg-hover); color: var(--text-primary); }
        .tab-btn.active { background: var(--accent); color: white; border-color: var(--accent); }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .card h2 {
            color: var(--accent);
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }
        
        .card h3 {
            color: var(--text-primary);
            margin: 15px 0 10px 0;
            font-size: 1.1em;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background: var(--bg-hover);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2.2em;
            font-weight: bold;
            color: var(--accent);
        }
        
        .metric-value.success { color: var(--success); }
        .metric-value.warning { color: var(--warning); }
        .metric-value.failure { color: var(--failure); }
        .metric-value.neutral { color: var(--neutral); }
        
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .progress-bar {
            background: var(--bg-dark);
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s;
        }
        
        .progress-fill.success { background: var(--success); }
        .progress-fill.warning { background: var(--warning); }
        .progress-fill.failure { background: var(--failure); }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            background: var(--bg-hover);
            color: var(--accent);
            font-weight: 600;
        }
        
        tr:hover { background: var(--bg-hover); }
        
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        
        .status-badge.success { background: rgba(76, 175, 80, 0.2); color: var(--success); }
        .status-badge.warning { background: rgba(255, 152, 0, 0.2); color: var(--warning); }
        .status-badge.failure { background: rgba(244, 67, 54, 0.2); color: var(--failure); }
        .status-badge.neutral { background: rgba(158, 158, 158, 0.2); color: var(--neutral); }
        
        .stage-pipeline {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
            margin: 15px 0;
        }
        
        .stage-node {
            background: var(--bg-hover);
            border: 2px solid var(--border);
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 0.85em;
            text-align: center;
            min-width: 100px;
        }
        
        .stage-node.success { border-color: var(--success); }
        .stage-node.failure { border-color: var(--failure); }
        .stage-node.pending { border-color: var(--neutral); opacity: 0.6; }
        
        .stage-arrow {
            color: var(--text-secondary);
            font-size: 1.2em;
        }
        
        .scenario-card {
            background: var(--bg-hover);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
        }
        
        .scenario-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: var(--bg-card);
            cursor: pointer;
            border-bottom: 1px solid var(--border);
        }
        
        .scenario-header:hover { background: var(--bg-hover); }
        
        .scenario-title {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .scenario-meta {
            display: flex;
            gap: 15px;
            color: var(--text-secondary);
            font-size: 0.9em;
        }
        
        .scenario-details {
            display: none;
            padding: 15px;
        }
        
        .scenario-card.expanded .scenario-details { display: block; }
        
        .scenario-images {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        
        .scenario-image {
            background: var(--bg-dark);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        
        .scenario-image img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        
        .scenario-image .caption {
            margin-top: 8px;
            font-size: 0.85em;
            color: var(--text-secondary);
        }
        
        .issue-list {
            list-style: none;
        }
        
        .issue-item {
            padding: 12px;
            margin: 8px 0;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .issue-item.error { background: rgba(244, 67, 54, 0.1); border-color: var(--failure); }
        .issue-item.warning { background: rgba(255, 152, 0, 0.1); border-color: var(--warning); }
        .issue-item.info { background: rgba(74, 144, 217, 0.1); border-color: var(--accent); }
        
        .issue-category {
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.85em;
            margin-bottom: 4px;
        }
        
        .issue-message { color: var(--text-primary); }
        
        .issue-suggestion {
            margin-top: 8px;
            padding: 8px;
            background: var(--bg-dark);
            border-radius: 4px;
            font-size: 0.9em;
            color: var(--text-secondary);
        }
        
        .bar-chart {
            margin: 15px 0;
        }
        
        .bar-row {
            display: flex;
            align-items: center;
            margin: 8px 0;
        }
        
        .bar-label {
            width: 200px;
            color: var(--text-secondary);
            font-size: 0.9em;
        }
        
        .bar-track {
            flex: 1;
            background: var(--bg-dark);
            height: 24px;
            border-radius: 4px;
            overflow: hidden;
            margin: 0 15px;
        }
        
        .bar-fill {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: 600;
            font-size: 0.85em;
            border-radius: 4px;
        }
        
        .bar-fill.success { background: var(--success); }
        .bar-fill.warning { background: var(--warning); }
        .bar-fill.failure { background: var(--failure); }
        .bar-fill.accent { background: var(--accent); }
        
        .bar-count {
            width: 60px;
            text-align: right;
            color: var(--text-secondary);
            font-size: 0.9em;
        }
        
        .repair-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 10px 0;
        }
        
        .repair-item {
            background: var(--bg-dark);
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }
        
        .repair-count {
            font-size: 1.5em;
            font-weight: bold;
            color: var(--warning);
        }
        
        .repair-type {
            font-size: 0.8em;
            color: var(--text-secondary);
            margin-top: 4px;
        }
        
        .timeline {
            position: relative;
            padding-left: 30px;
            margin: 20px 0;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--border);
        }
        
        .timeline-item {
            position: relative;
            margin: 15px 0;
            padding: 10px 15px;
            background: var(--bg-hover);
            border-radius: 8px;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 15px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent);
        }
        
        .timeline-item.success::before { background: var(--success); }
        .timeline-item.failure::before { background: var(--failure); }
        
        footer {
            text-align: center;
            padding: 30px 0;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
            margin-top: 30px;
        }
        
        .collapsible-trigger {
            cursor: pointer;
            user-select: none;
        }
        
        .collapsible-trigger::before {
            content: '▶ ';
            display: inline-block;
            transition: transform 0.2s;
        }
        
        .collapsible-trigger.expanded::before {
            transform: rotate(90deg);
        }
        
        .filter-bar {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        .filter-bar select, .filter-bar input {
            background: var(--bg-hover);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
        }
        
        .filter-bar select:focus, .filter-bar input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .kept-badge {
            background: linear-gradient(135deg, #4a90d9, #7c4dff);
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            margin-left: 8px;
        }
        
        @media (max-width: 768px) {
            .tabs { flex-direction: column; }
            .tab-btn { width: 100%; }
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
            .bar-label { width: 120px; }
        }
    """


def _get_js() -> str:
    """Get JavaScript for the report."""
    return """
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(btn.dataset.tab).classList.add('active');
            });
        });
        
        // Scenario expansion
        document.querySelectorAll('.scenario-header').forEach(header => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('expanded');
            });
        });
        
        // Collapsible sections
        document.querySelectorAll('.collapsible-trigger').forEach(trigger => {
            trigger.addEventListener('click', () => {
                trigger.classList.toggle('expanded');
                const content = trigger.nextElementSibling;
                if (content) {
                    content.style.display = trigger.classList.contains('expanded') ? 'block' : 'none';
                }
            });
        });
        
        // Category filter
        const categoryFilter = document.getElementById('category-filter');
        const statusFilter = document.getElementById('status-filter');
        
        function filterScenarios() {
            const category = categoryFilter ? categoryFilter.value : 'all';
            const status = statusFilter ? statusFilter.value : 'all';
            
            document.querySelectorAll('.scenario-card').forEach(card => {
                const cardCategory = card.dataset.category;
                const cardStatus = card.dataset.status;
                
                const categoryMatch = category === 'all' || cardCategory === category;
                const statusMatch = status === 'all' || cardStatus === status;
                
                card.style.display = categoryMatch && statusMatch ? 'block' : 'none';
            });
        }
        
        if (categoryFilter) categoryFilter.addEventListener('change', filterScenarios);
        if (statusFilter) statusFilter.addEventListener('change', filterScenarios);
    """


def _generate_overview_section(analysis: RunAnalysis) -> str:
    """Generate overview section HTML."""
    success_rate = analysis.overall_success_rate * 100
    success_class = "success" if success_rate >= 80 else "warning" if success_rate >= 50 else "failure"
    
    score_class = _get_score_class(analysis.avg_validation_score)
    
    html = f"""
    <div class="card">
        <h2>📈 Run Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{analysis.total_scenarios}</div>
                <div class="metric-label">Total Scenarios</div>
            </div>
            <div class="metric-card">
                <div class="metric-value success">{analysis.successful_scenarios}</div>
                <div class="metric-label">Successful</div>
            </div>
            <div class="metric-card">
                <div class="metric-value failure">{analysis.failed_scenarios}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {success_class}">{success_rate:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {score_class}">{analysis.avg_validation_score:.2f}</div>
                <div class="metric-label">Avg Validation Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value neutral">{_format_duration(analysis.total_duration_s)}</div>
                <div class="metric-label">Total Duration</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>🎯 Pipeline Stage Success Rates</h2>
        <div class="bar-chart">
    """
    
    # Stage success rates bar chart
    stage_order = [
        "scenario_generation",
        "repair_loops",
        "route_generation",
        "baseline_validation",
        "carla_simulation",
        "video_generation",
    ]
    
    for stage in stage_order:
        rate = analysis.stage_success_rates.get(stage, 0) * 100
        bar_class = "success" if rate >= 80 else "warning" if rate >= 50 else "failure"
        html += f"""
            <div class="bar-row">
                <div class="bar-label">{stage.replace('_', ' ').title()}</div>
                <div class="bar-track">
                    <div class="bar-fill {bar_class}" style="width: {rate}%">{rate:.0f}%</div>
                </div>
            </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    # Quick failure analysis
    if analysis.common_failure_stages:
        html += """
    <div class="card">
        <h2>🚨 Common Failure Points</h2>
        <div class="bar-chart">
        """
        
        max_count = max(c for _, c in analysis.common_failure_stages) if analysis.common_failure_stages else 1
        for stage, count in analysis.common_failure_stages[:5]:
            pct = (count / max_count) * 100
            html += f"""
            <div class="bar-row">
                <div class="bar-label">{stage.replace('_', ' ').title()}</div>
                <div class="bar-track">
                    <div class="bar-fill failure" style="width: {pct}%">{count}</div>
                </div>
                <div class="bar-count">{count} failures</div>
            </div>
            """
        
        html += """
        </div>
    </div>
        """
    
    return html


def _generate_categories_section(analysis: RunAnalysis) -> str:
    """Generate categories section HTML."""
    html = """
    <div class="card">
        <h2>📁 Category Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Total</th>
                    <th>Success</th>
                    <th>Failed</th>
                    <th>Success Rate</th>
                    <th>Avg Score</th>
                    <th>Avg Attempts</th>
                    <th>Kept Best</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for cat_name in sorted(analysis.category_summaries.keys()):
        summary = analysis.category_summaries[cat_name]
        success_rate = (summary.successful_runs / summary.total_runs * 100) if summary.total_runs else 0
        rate_class = "success" if success_rate >= 80 else "warning" if success_rate >= 50 else "failure"
        score_class = _get_score_class(summary.avg_validation_score)
        
        html += f"""
            <tr>
                <td><strong>{cat_name}</strong></td>
                <td>{summary.total_runs}</td>
                <td><span class="status-badge success">{summary.successful_runs}</span></td>
                <td><span class="status-badge failure">{summary.failed_runs}</span></td>
                <td><span class="status-badge {rate_class}">{success_rate:.0f}%</span></td>
                <td><span class="status-badge {score_class}">{summary.avg_validation_score:.2f}</span></td>
                <td>{summary.avg_attempts:.1f}</td>
                <td>{summary.kept_best_count}</td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    # Per-category details
    for cat_name in sorted(analysis.category_summaries.keys()):
        summary = analysis.category_summaries[cat_name]
        
        html += f"""
    <div class="card">
        <h2 class="collapsible-trigger">{cat_name}</h2>
        <div style="display: none;">
            <h3>Repair Breakdown</h3>
            <div class="repair-breakdown">
        """
        
        for repair_type, count in summary.repairs_by_type.items():
            html += f"""
                <div class="repair-item">
                    <div class="repair-count">{count}</div>
                    <div class="repair-type">{repair_type.replace('_', ' ')}</div>
                </div>
            """
        
        html += """
            </div>
        """
        
        if summary.failure_stage_counts:
            html += """
            <h3>Failure Stages</h3>
            <div class="bar-chart">
            """
            max_count = max(summary.failure_stage_counts.values()) if summary.failure_stage_counts else 1
            for stage, count in sorted(summary.failure_stage_counts.items(), key=lambda x: -x[1]):
                pct = (count / max_count) * 100
                html += f"""
                <div class="bar-row">
                    <div class="bar-label">{stage.replace('_', ' ')}</div>
                    <div class="bar-track">
                        <div class="bar-fill failure" style="width: {pct}%">{count}</div>
                    </div>
                </div>
                """
            html += """
            </div>
            """
        
        if summary.common_issues:
            html += """
            <h3>Common Issues</h3>
            <ul class="issue-list">
            """
            for issue, count in summary.common_issues:
                html += f"""
                <li class="issue-item warning">
                    <div class="issue-message">{issue} <strong>({count}x)</strong></div>
                </li>
                """
            html += """
            </ul>
            """
        
        html += """
        </div>
    </div>
        """
    
    return html


def _generate_scenarios_section(analysis: RunAnalysis) -> str:
    """Generate scenarios section HTML."""
    # Build filter options
    categories = sorted(set(s.category for s in analysis.scenarios))
    
    html = """
    <div class="card">
        <h2>🎬 Individual Scenarios</h2>
        <div class="filter-bar">
            <label>Category:</label>
            <select id="category-filter">
                <option value="all">All Categories</option>
    """
    
    for cat in categories:
        html += f'<option value="{cat}">{cat}</option>'
    
    html += """
            </select>
            <label>Status:</label>
            <select id="status-filter">
                <option value="all">All</option>
                <option value="success">Success</option>
                <option value="failure">Failed</option>
            </select>
        </div>
    </div>
    """
    
    # Sort scenarios: kept_best first, then by category and variant
    sorted_scenarios = sorted(
        analysis.scenarios,
        key=lambda s: (not s.kept_best, s.category, s.variant_index)
    )
    
    for scenario in sorted_scenarios:
        status = "success" if scenario.success else "failure"
        status_class = "success" if scenario.success else "failure"
        score_class = _get_score_class(scenario.validation_score)
        
        kept_badge = '<span class="kept-badge">★ KEPT</span>' if scenario.kept_best else ""
        rank_info = f'<span class="status-badge accent">Rank #{scenario.best_rank}</span>' if scenario.best_rank else ""
        
        html += f"""
    <div class="scenario-card" data-category="{scenario.category}" data-status="{status}">
        <div class="scenario-header">
            <div>
                <span class="scenario-title">{scenario.run_key}</span>
                {kept_badge}
            </div>
            <div class="scenario-meta">
                <span class="status-badge {status_class}">{status.upper()}</span>
                {rank_info}
                <span>Score: <strong class="{score_class}">{scenario.validation_score or 'N/A'}</strong></span>
                <span>Attempts: {scenario.total_attempts}</span>
                <span>{_format_duration(scenario.elapsed_s)}</span>
            </div>
        </div>
        <div class="scenario-details">
            <h3>Pipeline Stages</h3>
            <div class="stage-pipeline">
        """
        
        for i, stage in enumerate(scenario.stages):
            if not stage.started:
                stage_class = "pending"
            elif stage.success:
                stage_class = "success"
            else:
                stage_class = "failure"
            
            html += f"""
                <div class="stage-node {stage_class}">{stage.name.replace('_', ' ').title()}</div>
            """
            if i < len(scenario.stages) - 1:
                html += '<span class="stage-arrow">→</span>'
        
        html += """
            </div>
        """
        
        # Failure info
        if scenario.failure_state:
            html += f"""
            <div class="issue-item error">
                <div class="issue-category">Failure at: {scenario.failure_state}</div>
                <div class="issue-message">{scenario.failure_reason or 'Unknown error'}</div>
            </div>
            """
        
        # Baseline metrics
        if scenario.baseline_status:
            baseline_class = "success" if scenario.baseline_status == "accept" else "failure"
            html += f"""
            <h3>Baseline Validation</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {baseline_class}">{scenario.baseline_status}</div>
                    <div class="metric-label">Status</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{scenario.baseline_rc:.2f if scenario.baseline_rc else 'N/A'}</div>
                    <div class="metric-label">Route Completion</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{scenario.baseline_ds:.2f if scenario.baseline_ds else 'N/A'}</div>
                    <div class="metric-label">Driving Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'success' if scenario.baseline_near_miss else 'neutral'}">{scenario.baseline_near_miss or 'N/A'}</div>
                    <div class="metric-label">Near Miss</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{scenario.baseline_min_ttc:.2f if scenario.baseline_min_ttc and scenario.baseline_min_ttc < 100 else 'N/A'}</div>
                    <div class="metric-label">Min TTC</div>
                </div>
            </div>
            """
        
        # Repair breakdown
        total_repairs = (
            scenario.repair_outer_attempts +
            scenario.repair_schema_attempts +
            scenario.repair_object_stage1_json +
            scenario.repair_object_stage1_evidence +
            scenario.repair_object_stage2_json +
            scenario.repair_object_stage2_validation
        )
        
        if total_repairs > 0:
            html += """
            <h3>Repair Efforts</h3>
            <div class="repair-breakdown">
            """
            
            repairs = [
                ("Outer", scenario.repair_outer_attempts),
                ("Schema", scenario.repair_schema_attempts),
                ("Obj Stage1 JSON", scenario.repair_object_stage1_json),
                ("Obj Stage1 Evidence", scenario.repair_object_stage1_evidence),
                ("Obj Stage2 JSON", scenario.repair_object_stage2_json),
                ("Obj Stage2 Validation", scenario.repair_object_stage2_validation),
            ]
            
            for name, count in repairs:
                if count > 0:
                    html += f"""
                <div class="repair-item">
                    <div class="repair-count">{count}</div>
                    <div class="repair-type">{name}</div>
                </div>
                    """
            
            html += """
            </div>
            """
            
            if scenario.template_fallback:
                html += '<div class="issue-item warning"><div class="issue-message">Template fallback was used</div></div>'
        
        # Validation issues
        if scenario.validation_issues:
            html += """
            <h3>Validation Issues</h3>
            <ul class="issue-list">
            """
            
            for issue in scenario.validation_issues[:10]:
                issue_class = "error" if issue.severity == "error" else "warning" if issue.severity == "warning" else "info"
                html += f"""
                <li class="issue-item {issue_class}">
                    <div class="issue-category">{issue.category}</div>
                    <div class="issue-message">{issue.message}</div>
                """
                if issue.suggestion:
                    html += f'<div class="issue-suggestion">💡 {issue.suggestion}</div>'
                html += "</li>"
            
            if len(scenario.validation_issues) > 10:
                html += f'<li class="issue-item info"><div class="issue-message">... and {len(scenario.validation_issues) - 10} more issues</div></li>'
            
            html += """
            </ul>
            """
        
        # Attempt history timeline
        if scenario.attempt_history:
            html += """
            <h3>Attempt History</h3>
            <div class="timeline">
            """
            
            for attempt in scenario.attempt_history:
                attempt_status = attempt.get("status", "unknown")
                attempt_class = "success" if attempt_status == "success" else "failure"
                score = attempt.get("validation_score")
                score_str = f" (score: {score:.2f})" if score is not None else ""
                
                html += f"""
                <div class="timeline-item {attempt_class}">
                    <strong>Attempt {attempt.get('attempt', '?')}</strong>: {attempt_status}{score_str}
                """
                
                if attempt.get("spec_errors"):
                    html += f'<div class="issue-suggestion">Spec errors: {", ".join(attempt["spec_errors"][:3])}</div>'
                if attempt.get("pipeline_error"):
                    error_text = str(attempt["pipeline_error"])[:200]
                    html += f'<div class="issue-suggestion">Pipeline error: {error_text}</div>'
                
                html += """
                </div>
                """
            
            html += """
            </div>
            """
        
        # Images
        images = []
        if scenario.scene_png_path and Path(scenario.scene_png_path).exists():
            images.append(("Scene Objects", scenario.scene_png_path))
        
        # Check for additional visualizations in the run directory
        if scenario.run_dir:
            run_dir = Path(scenario.run_dir)
            pipeline_dir = run_dir / "pipeline"
            if pipeline_dir.exists():
                for attempt_dir in sorted(pipeline_dir.iterdir(), reverse=True):
                    if not attempt_dir.is_dir():
                        continue
                    
                    viz_files = [
                        ("Legal Paths", "legal_paths_all.png"),
                        ("Picked Paths", "picked_paths_viz.png"),
                        ("Refined Paths", "picked_paths_refined_viz.png"),
                    ]
                    
                    for label, fname in viz_files:
                        fpath = attempt_dir / fname
                        if fpath.exists() and str(fpath) not in [p for _, p in images]:
                            images.append((label, str(fpath)))
                    
                    break  # Only check latest attempt
        
        if images:
            html += """
            <h3>Visualizations</h3>
            <div class="scenario-images">
            """
            
            for label, path in images:
                data_uri = _embed_image(path)
                if data_uri:
                    html += f"""
                <div class="scenario-image">
                    <img src="{data_uri}" alt="{label}">
                    <div class="caption">{label}</div>
                </div>
                    """
            
            html += """
            </div>
            """
        
        # Artifact links
        artifacts = []
        if scenario.spec_json_path and Path(scenario.spec_json_path).exists():
            artifacts.append(("Scenario Spec", scenario.spec_json_path))
        if scenario.scene_json_path and Path(scenario.scene_json_path).exists():
            artifacts.append(("Scene Objects JSON", scenario.scene_json_path))
        if scenario.routes_dir and Path(scenario.routes_dir).exists():
            artifacts.append(("Routes Directory", scenario.routes_dir))
        if scenario.run_dir and Path(scenario.run_dir).exists():
            artifacts.append(("Run Directory", scenario.run_dir))
        
        if artifacts:
            html += """
            <h3>Artifacts</h3>
            <ul>
            """
            for label, path in artifacts:
                html += f'<li><code>{label}</code>: <code>{path}</code></li>'
            html += """
            </ul>
            """
        
        html += """
        </div>
    </div>
        """
    
    return html


def _generate_pipeline_section(analysis: RunAnalysis) -> str:
    """Generate pipeline health section HTML."""
    html = """
    <div class="card">
        <h2>🔧 Pipeline Health Overview</h2>
        <p>Analysis of pipeline stage performance and bottlenecks.</p>
        
        <h3>Stage Success Rates</h3>
        <div class="bar-chart">
    """
    
    stage_order = [
        "scenario_generation",
        "repair_loops",
        "route_generation",
        "baseline_validation",
        "carla_simulation",
        "video_generation",
    ]
    
    for stage in stage_order:
        rate = analysis.stage_success_rates.get(stage, 0) * 100
        bar_class = "success" if rate >= 80 else "warning" if rate >= 50 else "failure"
        html += f"""
            <div class="bar-row">
                <div class="bar-label">{stage.replace('_', ' ').title()}</div>
                <div class="bar-track">
                    <div class="bar-fill {bar_class}" style="width: {rate}%">{rate:.0f}%</div>
                </div>
            </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    # Repair effectiveness
    html += """
    <div class="card">
        <h2>🔄 Repair Loop Effectiveness</h2>
    """
    
    # Compute repair stats
    total_repairs = sum(
        s.repair_outer_attempts +
        s.repair_schema_attempts +
        s.repair_object_stage1_json +
        s.repair_object_stage1_evidence +
        s.repair_object_stage2_json +
        s.repair_object_stage2_validation
        for s in analysis.scenarios
    )
    
    scenarios_needing_repair = sum(
        1 for s in analysis.scenarios
        if (s.repair_outer_attempts + s.repair_schema_attempts + 
            s.repair_object_stage1_json + s.repair_object_stage1_evidence +
            s.repair_object_stage2_json + s.repair_object_stage2_validation) > 0
    )
    
    repaired_successfully = sum(
        1 for s in analysis.scenarios
        if s.success and (s.repair_outer_attempts + s.repair_schema_attempts +
            s.repair_object_stage1_json + s.repair_object_stage1_evidence +
            s.repair_object_stage2_json + s.repair_object_stage2_validation) > 0
    )
    
    repair_success_rate = (repaired_successfully / scenarios_needing_repair * 100) if scenarios_needing_repair else 0
    
    html += f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_repairs}</div>
                <div class="metric-label">Total Repairs</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{scenarios_needing_repair}</div>
                <div class="metric-label">Scenarios Needing Repair</div>
            </div>
            <div class="metric-card">
                <div class="metric-value success">{repaired_successfully}</div>
                <div class="metric-label">Successfully Repaired</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'success' if repair_success_rate >= 50 else 'failure'}">{repair_success_rate:.0f}%</div>
                <div class="metric-label">Repair Success Rate</div>
            </div>
        </div>
    """
    
    # Repair type breakdown
    repair_totals = defaultdict(int)
    for s in analysis.scenarios:
        repair_totals["outer"] += s.repair_outer_attempts
        repair_totals["schema"] += s.repair_schema_attempts
        repair_totals["object_stage1_json"] += s.repair_object_stage1_json
        repair_totals["object_stage1_evidence"] += s.repair_object_stage1_evidence
        repair_totals["object_stage2_json"] += s.repair_object_stage2_json
        repair_totals["object_stage2_validation"] += s.repair_object_stage2_validation
    
    if sum(repair_totals.values()) > 0:
        html += """
        <h3>Repair Type Distribution</h3>
        <div class="bar-chart">
        """
        
        max_count = max(repair_totals.values()) if repair_totals else 1
        for rtype, count in sorted(repair_totals.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = (count / max_count) * 100
                html += f"""
            <div class="bar-row">
                <div class="bar-label">{rtype.replace('_', ' ')}</div>
                <div class="bar-track">
                    <div class="bar-fill warning" style="width: {pct}%">{count}</div>
                </div>
                <div class="bar-count">{count}</div>
            </div>
                """
        
        html += """
        </div>
        """
    
    html += """
    </div>
    """
    
    # Template fallback usage
    template_fallbacks = sum(1 for s in analysis.scenarios if s.template_fallback)
    if template_fallbacks > 0:
        html += f"""
    <div class="card">
        <h2>📋 Template Fallback Usage</h2>
        <p><strong>{template_fallbacks}</strong> scenarios used template fallback (LLM generation failed).</p>
    </div>
        """
    
    return html


def _generate_issues_section(analysis: RunAnalysis) -> str:
    """Generate issues section HTML."""
    html = """
    <div class="card">
        <h2>⚠️ Issue Summary</h2>
    """
    
    if analysis.issue_severity_counts:
        html += """
        <h3>Issue Severity Distribution</h3>
        <div class="metrics-grid">
        """
        
        for severity, count in sorted(analysis.issue_severity_counts.items()):
            sev_class = "failure" if severity == "error" else "warning" if severity == "warning" else "neutral"
            html += f"""
            <div class="metric-card">
                <div class="metric-value {sev_class}">{count}</div>
                <div class="metric-label">{severity.title()}</div>
            </div>
            """
        
        html += """
        </div>
        """
    
    html += """
    </div>
    """
    
    # Common validation issues
    if analysis.common_validation_issues:
        html += """
    <div class="card">
        <h2>🔍 Most Common Validation Issues</h2>
        <div class="bar-chart">
        """
        
        max_count = max(c for _, c in analysis.common_validation_issues) if analysis.common_validation_issues else 1
        for issue, count in analysis.common_validation_issues[:15]:
            pct = (count / max_count) * 100
            html += f"""
            <div class="bar-row">
                <div class="bar-label" title="{issue}">{issue[:40]}...</div>
                <div class="bar-track">
                    <div class="bar-fill warning" style="width: {pct}%">{count}</div>
                </div>
                <div class="bar-count">{count}x</div>
            </div>
            """
        
        html += """
        </div>
    </div>
        """
    
    # Common failure reasons
    if analysis.common_failure_reasons:
        html += """
    <div class="card">
        <h2>💥 Common Failure Reasons</h2>
        <ul class="issue-list">
        """
        
        for reason, count in analysis.common_failure_reasons[:10]:
            html += f"""
            <li class="issue-item error">
                <div class="issue-message">{reason} <strong>({count}x)</strong></div>
            </li>
            """
        
        html += """
        </ul>
    </div>
        """
    
    return html


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate debug visualization report for audit benchmark runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path, help="Path to the audit run directory")
    parser.add_argument("--output", "-o", type=Path, help="Output HTML file path (default: <run_dir>/debug_report.html)")
    parser.add_argument("--open", action="store_true", help="Open report in browser after generation")
    args = parser.parse_args()
    
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    print(f"Loading analysis from: {run_dir}")
    analysis = load_run_analysis(run_dir)
    
    print(f"Found {analysis.total_scenarios} scenarios across {len(analysis.category_summaries)} categories")
    print(f"Success rate: {analysis.overall_success_rate*100:.1f}%")
    print(f"Avg validation score: {analysis.avg_validation_score:.2f}")
    
    output_path = args.output or (run_dir / "debug_report.html")
    
    print(f"Generating HTML report...")
    generate_html_report(analysis, output_path)
    
    print(f"✓ Report saved to: {output_path}")
    
    if args.open:
        webbrowser.open(f"file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
