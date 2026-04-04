import argparse
import csv
import json
import math
import os
import shutil
import statistics
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate


ROUTE_PREFIXES = {
    "Interdrive_no_npc_": "No NPC",
    "Interdrive_npc_": "NPC",
}
DEFAULT_MODE_LABEL = "Custom"
MODE_ORDER = {"No NPC": 0, "NPC": 1, DEFAULT_MODE_LABEL: 2}

BASE_TYPE_LABELS: Dict[str, str] = {
    "hw": "Highway",
    "ins": "Intersection",
}

SCENARIO_DESCRIPTIONS: Dict[Tuple[str, str], str] = {
    ("ins", "ss"): "Cross: Straight-Straight",
    ("ins", "sl"): "Cross: Straight-Left",
    ("ins", "sr"): "Cross: Straight-Right",
    ("ins", "oppo"): "Cross: Opposite Lane",
    ("ins", "chaos"): "Cross: Chaos",
    ("ins", "crosschange"): "Change: Right-Straight",
    ("ins", "c"): "Change: Right-Straight",
    ("ins", "rl"): "Change: Right-Left",
    ("hw", "merge"): "Merge: Neighbor Lane",
    ("hw", "c"): "Change: Highway",
}

_ROUTE_AGENT_SPEC = (
    '("r1_town05_ins_c:2" "r2_town05_ins_c:2" "r3_town05_ins_c:2" '
    '"r4_town06_ins_c:2" "r5_town06_ins_c:2" "r6_town07_ins_c:2" '
    '"r7_town05_ins_ss:2" "r8_town05_ins_ss:2" "r9_town06_ins_ss:2" '
    '"r10_town07_ins_ss:2" "r11_town05_ins_sl:2" "r12_town06_ins_sl:2" '
    '"r13_town05_ins_sl:2" "r14_town07_ins_sl:2" "r15_town07_ins_sl:2" '
    '"r16_town05_ins_sl:2" "r17_town05_ins_sr:2" "r18_town05_ins_sr:2" '
    '"r19_town05_ins_sr:2" "r20_town06_ins_sr:2" "r21_town07_ins_sr:2" '
    '"r22_town07_ins_sr:2" "r23_town05_ins_oppo:3" "r24_town05_ins_rl:3" '
    '"r25_town05_ins_crosschange:3" "r26_town05_ins_chaos:6" '
    '"r27_town06_hw_merge:3" "r28_town06_hw_c:6" "r29_town06_hw_merge:4" '
    '"r30_town06_hw_merge:4" "r31_town05_ins_oppo:4" "r32_town05_ins_oppo:4" '
    '"r33_town05_ins_rl:4" "r34_town05_ins_rl:4" "r35_town05_ins_crosschange:4" '
    '"r36_town05_ins_crosschange:4" "r37_town05_ins_chaos:8" '
    '"r38_town05_ins_chaos:8" "r39_town06_hw_c:8" "r40_town06_hw_c:8" '
    '"r41_town05_ins_oppo:4" "r42_town05_ins_rl:4" "r43_town05_ins_crosschange:4" '
    '"r44_town05_ins_chaos:8" "r45_town06_hw_merge:4" "r46_town06_hw_c:7")'
)


def parse_route_agent_map(spec: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    tokens = spec.replace("(", "").replace(")", "").replace('"', "").split()
    for token in tokens:
        if ":" not in token:
            continue
        route, count = token.split(":", 1)
        if not count.isdigit():
            continue
        mapping[route] = int(count)
    return mapping


ROUTE_AGENT_MAP = parse_route_agent_map(_ROUTE_AGENT_SPEC)


def detect_mode_label(name: str) -> Tuple[str, str]:
    for prefix, mode in ROUTE_PREFIXES.items():
        if name.startswith(prefix):
            return mode, prefix
    return DEFAULT_MODE_LABEL, ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def scenario_type_label(base_type: str, sub_type: str) -> str:
    base_label = BASE_TYPE_LABELS.get(base_type, base_type.title() if base_type else "Unknown")
    description = SCENARIO_DESCRIPTIONS.get((base_type, sub_type))
    if not description:
        description = sub_type.replace("_", " ").title() if sub_type else base_label
    if description.lower().startswith(base_label.lower()):
        return description
    return f"{base_label} - {description}"


def scenario_bar_label(route_id: str, agent_count: Optional[int], base_type: str, sub_type: str) -> str:
    parts = route_id.split("_")
    route_code = parts[0] if parts else route_id
    town = parts[1] if len(parts) > 1 else ""
    type_label = scenario_type_label(base_type, sub_type)
    town_segment = f"{town} - " if town else ""
    agents_text = f"{agent_count} agents" if agent_count is not None else "agents: n/a"
    return f"{route_code} ({agents_text})\n{town_segment}{type_label}"


@dataclass
class ScenarioMeta:
    scenario_name: str
    route_id: str
    agent_count: Optional[int]
    base_type: str
    sub_type: str


@dataclass(frozen=True)
class NegotiationLogSource:
    scenario_name: str
    source: Path
    source_count: int


def derive_scenario_meta(scenario_name: str) -> ScenarioMeta:
    try:
        idx = scenario_name.index("_r")
    except ValueError as exc:
        raise ValueError(f"Scenario name not in expected format: {scenario_name}") from exc
    route_id = scenario_name[idx + 1 :]
    parts = route_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Route id does not contain expected parts: {route_id}")
    base_type = parts[2]
    sub_type = "_".join(parts[3:]) if len(parts) > 3 else ""
    agent_count = ROUTE_AGENT_MAP.get(route_id)
    return ScenarioMeta(
        scenario_name=scenario_name,
        route_id=route_id,
        agent_count=agent_count,
        base_type=base_type,
        sub_type=sub_type,
    )


def iter_negotiations(data: Dict) -> Sequence[Tuple[str, str, int, Dict]]:
    records = []
    for stage_key, stage_val in data.items():
        if stage_key == "action" or not isinstance(stage_val, dict):
            continue
        try:
            timestamp = int(stage_key)
        except ValueError:
            timestamp = -1
        for event_key, payload in stage_val.items():
            if not isinstance(payload, dict):
                continue
            records.append((stage_key, event_key, timestamp, payload))
    return records


def extract_message_features(payload: Dict) -> Dict[str, Optional[float]]:
    content = payload.get("content") or []
    messages = [msg for msg in content if isinstance(msg, dict)]
    texts = [msg.get("message", "") for msg in messages if isinstance(msg.get("message"), str)]
    rounds = len(texts)
    if rounds:
        word_counts = [len(t.split()) for t in texts]
        char_counts = [len(t) for t in texts]
        avg_words = statistics.mean(word_counts)
        avg_chars = statistics.mean(char_counts)
    else:
        avg_words = 0.0
        avg_chars = 0.0
    unique_agents = {
        msg.get("id") for msg in messages if isinstance(msg.get("id"), (int, str))
    }
    return {
        "communication_rounds": rounds,
        "unique_speakers": len(unique_agents),
        "avg_words_per_message": float(avg_words),
        "avg_chars_per_message": float(avg_chars),
        "speaker_ids": unique_agents,
    }


def negotiation_overall_statistics(df: pd.DataFrame) -> Dict[str, float]:
    rounds = df["rounds"]
    return {
        "total_negotiations": int(len(rounds)),
        "total_rounds": int(rounds.sum()),
        "mean_rounds": float(rounds.mean()) if len(rounds) else 0.0,
        "median_rounds": float(rounds.median()) if len(rounds) else 0.0,
        "std_rounds": float(rounds.std(ddof=1)) if len(rounds) > 1 else 0.0,
        "min_rounds": float(rounds.min()) if len(rounds) else 0.0,
        "max_rounds": float(rounds.max()) if len(rounds) else 0.0,
    }


def negotiation_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(
            ["scenario", "route_id", "mode", "agent_count", "base_type", "sub_type", "scenario_type_label"],
            dropna=False,
        )
        .agg(
            negotiations=("rounds", "size"),
            total_rounds=("rounds", "sum"),
            mean_rounds=("rounds", "mean"),
            median_rounds=("rounds", "median"),
            std_rounds=("rounds", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            min_rounds=("rounds", "min"),
            max_rounds=("rounds", "max"),
            avg_words=("avg_words_per_message", "mean"),
            avg_chars=("avg_chars_per_message", "mean"),
            suggestions=("suggestion", "sum"),
        )
        .reset_index()
    )
    summary["scenario_bar_label"] = summary.apply(
        lambda row: scenario_bar_label(row["route_id"], row["agent_count"], row["base_type"], row["sub_type"]),
        axis=1,
    )
    return summary


def negotiation_group_summary(df: pd.DataFrame, group_field: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(group_field)
        .agg(
            negotiations=("rounds", "size"),
            total_rounds=("rounds", "sum"),
            mean_rounds=("rounds", "mean"),
            median_rounds=("rounds", "median"),
            std_rounds=("rounds", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            min_rounds=("rounds", "min"),
            max_rounds=("rounds", "max"),
            avg_words=("avg_words_per_message", "mean"),
            avg_chars=("avg_chars_per_message", "mean"),
            suggestions=("suggestion", "sum"),
        )
        .reset_index()
    )
    return summary


def negotiation_flavor_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(["base_type", "sub_type", "scenario_type_label"])
        .agg(
            negotiations=("rounds", "size"),
            total_rounds=("rounds", "sum"),
            mean_rounds=("rounds", "mean"),
            median_rounds=("rounds", "median"),
            std_rounds=("rounds", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
            min_rounds=("rounds", "min"),
            max_rounds=("rounds", "max"),
            avg_words=("avg_words_per_message", "mean"),
            avg_chars=("avg_chars_per_message", "mean"),
            suggestions=("suggestion", "sum"),
        )
        .reset_index()
        .sort_values(["base_type", "sub_type"])
    )
    return summary


def save_negotiation_tables(
    df: pd.DataFrame,
    scenario_stats: pd.DataFrame,
    agent_stats: pd.DataFrame,
    base_stats: pd.DataFrame,
    flavor_stats: pd.DataFrame,
    tables_dir: Path,
) -> Dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    detailed_df = df.copy()
    detailed_df.to_csv(tables_dir / "negotiations_detailed_full.csv", index=False)
    outputs["detailed"] = tables_dir / "negotiations_detailed_full.csv"

    scenario_stats.to_csv(tables_dir / "negotiations_scenario_summary.csv", index=False)
    outputs["scenario"] = tables_dir / "negotiations_scenario_summary.csv"

    agent_stats.to_csv(tables_dir / "negotiations_by_agent_count.csv", index=False)
    outputs["agent"] = tables_dir / "negotiations_by_agent_count.csv"

    base_stats.to_csv(tables_dir / "negotiations_by_base_type.csv", index=False)
    outputs["base"] = tables_dir / "negotiations_by_base_type.csv"

    flavor_stats.to_csv(tables_dir / "negotiations_by_setting.csv", index=False)
    outputs["flavor"] = tables_dir / "negotiations_by_setting.csv"

    return outputs


def save_negotiation_figures(
    df: pd.DataFrame,
    scenario_stats: pd.DataFrame,
    figures_dir: Path,
) -> Dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    figure_paths: Dict[str, Path] = {}

    if not scenario_stats.empty:
        ordered = scenario_stats.sort_values("mean_rounds", ascending=False)
        fig1_height = max(10, 0.35 * len(ordered))
        fig1, ax1 = plt.subplots(figsize=(14, fig1_height))
        sns.barplot(
            data=ordered,
            y="scenario_bar_label",
            x="mean_rounds",
            color="#1f77b4",
            ax=ax1,
        )
        ax1.set_title("Average Negotiation Rounds per Scenario")
        ax1.set_xlabel("Average rounds")
        ax1.set_ylabel("Scenario")
        ax1.grid(axis="x", alpha=0.3)
        max_rounds = ordered["mean_rounds"].max() if not ordered.empty else 0
        ax1.set_xlim(0, max_rounds + 3)
        for patch, mean_value, negotiations in zip(
            ax1.patches, ordered["mean_rounds"], ordered["negotiations"]
        ):
            width = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax1.text(
                width + 0.3,
                y,
                f"{mean_value:.1f} avg\n{int(negotiations)} negotiations",
                va="center",
                fontsize=9,
            )
        fig1.tight_layout(rect=(0, 0.02, 1, 1))
        path = figures_dir / "negotiations_avg_rounds_per_scenario.png"
        fig1.savefig(path)
        plt.close(fig1)
        figure_paths["avg_rounds_per_scenario"] = path

    if not df.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df, x="agent_count", y="rounds", ax=ax2)
        sns.stripplot(
            data=df,
            x="agent_count",
            y="rounds",
            color="black",
            alpha=0.4,
            ax=ax2,
            dodge=True,
        )
        ax2.set_title("Negotiation Rounds by Agent Count")
        ax2.set_xlabel("Agent Count")
        ax2.set_ylabel("Rounds")
        fig2.tight_layout()
        path = figures_dir / "negotiations_rounds_by_agent_count.png"
        fig2.savefig(path)
        plt.close(fig2)
        figure_paths["rounds_by_agent_count"] = path

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        order = (
            df.groupby("scenario_type_label")["rounds"]
            .mean()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(
            data=df,
            x="scenario_type_label",
            y="rounds",
            order=order,
            showfliers=False,
            ax=ax3,
        )
        ax3.set_title("Negotiation Rounds by Scenario Setting")
        ax3.set_xlabel("Scenario Setting")
        ax3.set_ylabel("Rounds")
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right")
        fig3.tight_layout()
        path = figures_dir / "negotiations_rounds_by_setting.png"
        fig3.savefig(path)
        plt.close(fig3)
        figure_paths["rounds_by_setting"] = path

        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=scenario_stats,
            x="agent_count",
            y="mean_rounds",
            hue="scenario_type_label",
            size="negotiations",
            sizes=(60, 360),
            ax=ax4,
        )
        ax4.set_title("Average Rounds vs Agent Count")
        ax4.set_xlabel("Agent Count")
        ax4.set_ylabel("Average Rounds")
        legend = ax4.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            title="Scenario type\nMarker size = # negotiations",
            borderaxespad=0,
        )
        if legend:
            for text in legend.texts:
                label = text.get_text()
                if label.strip().isdigit():
                    text.set_text(f"{label.strip()} negotiations")
        fig4.tight_layout()
        path = figures_dir / "negotiations_rounds_vs_agent_count.png"
        fig4.savefig(path)
        plt.close(fig4)
        figure_paths["rounds_vs_agent_count"] = path

    return figure_paths


def negotiation_flavors_to_string(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display_df = df.copy()
    columns = []
    if "scenario_type_label" in display_df.columns:
        columns.append("scenario_type_label")
    columns.extend(col for col in ["base_type", "sub_type"] if col in display_df.columns)
    stat_columns = [
        "negotiations",
        "total_rounds",
        "mean_rounds",
        "median_rounds",
        "std_rounds",
        "min_rounds",
        "max_rounds",
        "avg_words",
        "avg_chars",
        "suggestions",
    ]
    columns.extend(col for col in stat_columns if col in display_df.columns)
    if columns:
        display_df = display_df[columns]
    return display_df.to_string(index=False)


def render_negotiation_report(
    overall_stats: Dict[str, float],
    agent_stats: pd.DataFrame,
    base_stats: pd.DataFrame,
    flavor_stats: pd.DataFrame,
) -> str:
    lines = []
    lines.append("=== Negotiation Overview ===")
    lines.append(
        textwrap.dedent(
            f"""
Total negotiations: {overall_stats.get('total_negotiations', 0)}
Total rounds: {overall_stats.get('total_rounds', 0)}
Mean rounds: {overall_stats.get('mean_rounds', 0.0):.2f}
Median rounds: {overall_stats.get('median_rounds', 0.0):.2f}
Std dev rounds: {overall_stats.get('std_rounds', 0.0):.2f}
Min rounds: {overall_stats.get('min_rounds', 0.0):.0f}
Max rounds: {overall_stats.get('max_rounds', 0.0):.0f}
""".strip()
        )
    )
    lines.append("")

    if not agent_stats.empty:
        lines.append("=== Breakdown by Agent Count ===")
        lines.append(agent_stats.to_string(index=False))
        lines.append("")

    if not base_stats.empty:
        base_display = base_stats.copy()
        if "base_type" in base_display.columns:
            base_display["base_label"] = base_display["base_type"].map(
                lambda base: BASE_TYPE_LABELS.get(base, base.title())
            )
        base_columns = [
            "base_label" if "base_label" in base_display.columns else "base_type",
            "negotiations",
            "total_rounds",
            "mean_rounds",
            "median_rounds",
            "std_rounds",
            "min_rounds",
            "max_rounds",
        ]
        base_display = base_display[[col for col in base_columns if col in base_display.columns]]
        lines.append("=== Breakdown by Base Scenario Type ===")
        lines.append(base_display.to_string(index=False))
        lines.append("")

    if not flavor_stats.empty:
        lines.append("=== Breakdown by Scenario Setting (base + subtype) ===")
        lines.append(negotiation_flavors_to_string(flavor_stats))
        legend_entries: List[str] = []
        for (base, sub), desc in sorted(
            SCENARIO_DESCRIPTIONS.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            base_label = BASE_TYPE_LABELS.get(base, base.title())
            entry = f"  {base_label} - {desc}"
            if entry not in legend_entries:
                legend_entries.append(entry)
        if legend_entries:
            lines.append("")
            lines.append("Scenario Type Legend:")
            lines.extend(legend_entries)

    return "\n".join(lines).strip()


def average(values):
    return sum(values) / len(values) if values else 0.0


def extract_route_index(route_name: str) -> int:
    head = route_name.split("_", 1)[0]
    if head.startswith("r"):
        number = head[1:]
        if number.isdigit():
            return int(number)
    return sys.maxsize


def route_entry_sort_key(entry):
    return (
        extract_route_index(entry["route_name"]),
        entry["route_name"].lower(),
        MODE_ORDER.get(entry["mode"], len(MODE_ORDER)),
    )


def load_ego_records(route_path: str):
    records = []
    for ego_dir in sorted(d for d in os.listdir(route_path) if d.startswith("ego_vehicle")):
        result_path = os.path.join(route_path, ego_dir, "results.json")
        data = safe_load_json(result_path)
        if not data:
            continue
        ego_records = data.get("_checkpoint", {}).get("records", [])
        if ego_records:
            records.append(ego_records[-1])
    return records


def aggregate_route_records(records):
    score_route_vals = []
    score_penalty_vals = []
    score_composed_vals = []
    success_flags = []
    duration_game_vals = []
    duration_system_vals = []
    infractions_totals = defaultdict(int)
    hug_score_vals = []
    hug_route_completion_vals = []
    hug_epdm_vals = []

    for record in records:
        scores = record.get("scores", {})
        meta = record.get("meta", {})

        score_route_vals.append(scores.get("score_route", 0.0))
        score_penalty_vals.append(scores.get("score_penalty", 0.0))
        score_composed_vals.append(scores.get("score_composed", 0.0))
        duration_game_vals.append(meta.get("duration_game", 0.0))
        duration_system_vals.append(meta.get("duration_system", 0.0))
        success_flags.append(1 if scores.get("score_composed", 0.0) > 99.95 else 0)

        hugsim = record.get("hugsim", {})
        if hugsim.get("available", False):
            hug_score_vals.append(float(hugsim.get("hug_score", 0.0)))
            hug_route_completion_vals.append(float(hugsim.get("route_completion", 0.0)))
            hug_epdm_vals.append(float(hugsim.get("mean_epdm_score", 0.0)))

        infractions = record.get("infractions", {})
        if infractions:
            for key, events in infractions.items():
                if isinstance(events, list):
                    if events:
                        infractions_totals[key] += len(events)
                elif isinstance(events, (int, float)):
                    infractions_totals[key] += events
                elif isinstance(events, dict):
                    infractions_totals[key] += len(events)

    aggregated = {
        "score_route": average(score_route_vals),
        "score_penalty": average(score_penalty_vals),
        "score_composed": average(score_composed_vals),
        "success_flag": average(success_flags),
        "duration_game": average(duration_game_vals),
        "duration_system": average(duration_system_vals),
        "hug_score": average(hug_score_vals) if hug_score_vals else None,
        "hug_route_completion": average(hug_route_completion_vals) if hug_route_completion_vals else None,
        "hug_epdm": average(hug_epdm_vals) if hug_epdm_vals else None,
    }
    infractions_clean = {k: v for k, v in infractions_totals.items() if v > 0}
    if infractions_clean:
        aggregated["infractions"] = infractions_clean
    aggregated["ego_runs"] = len(records)
    return aggregated


def load_negotiation_logs(base_path: str):
    stats: Dict[str, Dict] = {}
    detailed_records: List[Dict] = []
    log_sources: List[NegotiationLogSource] = []

    base_path_obj = Path(base_path)
    image_root = base_path_obj.parent / "image" / base_path_obj.name
    candidate_roots: List[Path] = []
    if image_root.is_dir():
        candidate_roots.append(image_root)

    legacy_logs = base_path_obj / "collected_nego_logs"
    if legacy_logs.is_dir():
        candidate_roots.append(legacy_logs)

    if not candidate_roots:
        return stats, detailed_records, log_sources

    seen_scenarios: set[str] = set()

    for root in candidate_roots:
        for entry in sorted(root.iterdir()):
            if entry.name.startswith("."):
                continue

            scenario_name = entry.stem if entry.is_file() and entry.suffix == ".json" else entry.name
            if scenario_name in seen_scenarios:
                continue
            seen_scenarios.add(scenario_name)

            try:
                meta = derive_scenario_meta(scenario_name)
            except ValueError:
                meta = ScenarioMeta(
                    scenario_name=scenario_name,
                    route_id=scenario_name,
                    agent_count=None,
                    base_type="",
                    sub_type="",
                )

            run_logs: List[Tuple[str, Path]] = []
            if entry.is_dir():
                for run_dir in sorted(entry.iterdir()):
                    log_path = run_dir / "log" / "nego.json"
                    if log_path.is_file():
                        run_logs.append((run_dir.name, log_path))
            elif entry.suffix == ".json":
                run_logs.append(("", entry))

            if not run_logs:
                continue

            selected_source = run_logs[-1][1]
            mode_label, _ = detect_mode_label(scenario_name)
            log_sources.append(
                NegotiationLogSource(
                    scenario_name=scenario_name,
                    source=selected_source,
                    source_count=len(run_logs),
                )
            )

            summary = stats.setdefault(
                scenario_name,
                {
                    "count": 0,
                    "rounds": [],
                    "suggestions": 0,
                    "unique_speakers": set(),
                    "cons_scores": [],
                    "safety_scores": [],
                    "efficiency_scores": [],
                    "total_scores": [],
                    "min_distances": [],
                    "avg_words": [],
                    "avg_chars": [],
                    "base_type": meta.base_type,
                    "sub_type": meta.sub_type,
                    "mode": mode_label,
                    "agent_count": meta.agent_count,
                    "route_id": meta.route_id,
                    "scenario_type_label": scenario_type_label(meta.base_type, meta.sub_type),
                    "scenario_bar_label": scenario_bar_label(
                        meta.route_id, meta.agent_count, meta.base_type, meta.sub_type
                    ),
                },
            )

            for run_id, log_path in run_logs:
                data = safe_load_json(log_path)
                if not data:
                    continue
                for stage_key, event_key, timestamp, payload in iter_negotiations(data):
                    features = extract_message_features(payload)
                    if features["communication_rounds"] <= 0:
                        continue
                    suggestion_flag = payload.get("suggestion") is not None

                    summary["count"] += 1
                    summary["rounds"].append(features["communication_rounds"])
                    summary["suggestions"] += 1 if suggestion_flag else 0
                    summary["unique_speakers"].update(features["speaker_ids"])
                    summary["avg_words"].append(features["avg_words_per_message"])
                    summary["avg_chars"].append(features["avg_chars_per_message"])

                    for key, bucket in (
                        ("cons_score", "cons_scores"),
                        ("safety_score", "safety_scores"),
                        ("efficiency_score", "efficiency_scores"),
                        ("total_score", "total_scores"),
                        ("min_distance", "min_distances"),
                    ):
                        value = payload.get(key)
                        if isinstance(value, (int, float)):
                            summary[bucket].append(float(value))

                    detailed_records.append(
                        {
                            "scenario": scenario_name,
                            "route_id": meta.route_id,
                            "mode": mode_label,
                            "run_id": run_id,
                            "stage": stage_key,
                            "event": event_key,
                            "timestamp": timestamp,
                            "pair_key": event_key,
                            "rounds": features["communication_rounds"],
                            "unique_speakers": features["unique_speakers"],
                            "avg_words_per_message": features["avg_words_per_message"],
                            "avg_chars_per_message": features["avg_chars_per_message"],
                            "suggestion": suggestion_flag,
                            "cons_score": payload.get("cons_score"),
                            "safety_score": payload.get("safety_score"),
                            "efficiency_score": payload.get("efficiency_score"),
                            "total_score": payload.get("total_score"),
                            "min_distance": payload.get("min_distance"),
                            "base_type": meta.base_type,
                            "sub_type": meta.sub_type,
                            "scenario_type_label": scenario_type_label(meta.base_type, meta.sub_type),
                            "agent_count": meta.agent_count,
                        }
                    )

    for route_key, summary in stats.items():
        rounds = summary["rounds"]
        summary["total_rounds"] = sum(rounds)
        summary["avg_rounds"] = sum(rounds) / summary["count"] if summary["count"] else 0.0
        summary["median_rounds"] = statistics.median(rounds) if rounds else 0.0
        summary["max_rounds"] = max(rounds) if rounds else 0
        summary["unique_speakers"] = len(summary["unique_speakers"])
        summary["avg_consensus"] = average(summary["cons_scores"])
        summary["avg_safety"] = average(summary["safety_scores"])
        summary["avg_efficiency"] = average(summary["efficiency_scores"])
        summary["avg_total_score"] = average(summary["total_scores"])
        summary["avg_min_distance"] = average(summary["min_distances"])
        summary["avg_words_per_message"] = average(summary["avg_words"])
        summary["avg_chars_per_message"] = average(summary["avg_chars"])

    return stats, detailed_records, log_sources


def analyze_results(main_path: str):
    discovered_routes = []
    for route_dir in sorted(os.listdir(main_path)):
        route_path = os.path.join(main_path, route_dir)
        if not os.path.isdir(route_path):
            continue
        mode, prefix = detect_mode_label(route_dir)
        discovered_routes.append((mode, prefix, route_dir))
    if not discovered_routes:
        return None

    route_entries = []
    infractions_summary = []
    unmatched_routes = []
    total_ego_runs = 0
    infraction_totals = defaultdict(int)
    negotiation_stats, negotiation_records, negotiation_sources = load_negotiation_logs(main_path)
    negotiation_rounds_all: List[int] = []
    negotiation_cons_scores_all: List[float] = []
    negotiation_safety_scores_all: List[float] = []
    negotiation_efficiency_scores_all: List[float] = []
    negotiation_total_scores_all: List[float] = []
    negotiation_min_distances_all: List[float] = []
    negotiation_suggestions_total = 0
    total_negotiations = 0

    for mode, prefix, route_dir in sorted(discovered_routes, key=lambda x: x[2]):
        route_path = os.path.join(main_path, route_dir)
        records = load_ego_records(route_path)
        if not records:
            continue

        agent_count = len([d for d in os.listdir(route_path) if d.startswith("ego_vehicle")])

        route_name = route_dir.replace(prefix, "", 1)

        aggregated = aggregate_route_records(records)
        score_route = aggregated["score_route"]
        score_penalty = aggregated["score_penalty"]
        score_composed = aggregated["score_composed"]
        duration_game = aggregated["duration_game"]
        duration_system = aggregated["duration_system"]
        success_flag = aggregated["success_flag"]
        total_ego_runs += aggregated["ego_runs"]

        infractions = aggregated.get("infractions")
        if infractions:
            for key, count in infractions.items():
                infraction_totals[key] += count
            infractions_summary.append(
                {
                    "route_name": route_name,
                    "mode": mode,
                    "details": infractions,
                }
            )

        negotiation_info = negotiation_stats.get(route_dir, {})
        negotiation_count = negotiation_info.get("count", 0)
        negotiation_rounds = negotiation_info.get("rounds", [])
        negotiation_avg_rounds = negotiation_info.get("avg_rounds", 0.0)
        negotiation_median_rounds = negotiation_info.get("median_rounds", 0.0)
        negotiation_max_rounds = negotiation_info.get("max_rounds", 0)
        negotiation_suggestions = negotiation_info.get("suggestions", 0)
        negotiation_speakers = negotiation_info.get("unique_speakers", 0)
        negotiation_cons_scores = negotiation_info.get("cons_scores", [])
        negotiation_safety_scores = negotiation_info.get("safety_scores", [])
        negotiation_efficiency_scores = negotiation_info.get("efficiency_scores", [])
        negotiation_total_scores = negotiation_info.get("total_scores", [])
        negotiation_min_distances = negotiation_info.get("min_distances", [])
        negotiation_avg_consensus = average(negotiation_cons_scores)
        negotiation_avg_safety = average(negotiation_safety_scores)
        negotiation_avg_efficiency = average(negotiation_efficiency_scores)
        negotiation_avg_total = average(negotiation_total_scores)
        negotiation_avg_min_distance = average(negotiation_min_distances)
        if negotiation_info:
            base_type = negotiation_info.get("base_type", "")
            sub_type = negotiation_info.get("sub_type", "")
            route_id_meta = negotiation_info.get("route_id", route_dir)
            negotiation_info["agent_count"] = agent_count
            negotiation_info["scenario_type_label"] = scenario_type_label(base_type, sub_type)
            negotiation_info["scenario_bar_label"] = scenario_bar_label(
                route_id_meta, agent_count, base_type, sub_type
            )

        if negotiation_count:
            total_negotiations += negotiation_count
            negotiation_rounds_all.extend(negotiation_rounds)
            negotiation_cons_scores_all.extend(negotiation_cons_scores)
            negotiation_safety_scores_all.extend(negotiation_safety_scores)
            negotiation_efficiency_scores_all.extend(negotiation_efficiency_scores)
            negotiation_total_scores_all.extend(negotiation_total_scores)
            negotiation_min_distances_all.extend(negotiation_min_distances)
            negotiation_suggestions_total += negotiation_suggestions

        route_entries.append(
            {
                "route_id": route_dir,
                "route_name": route_name,
                "mode": mode,
                "score_composed": score_composed,
                "score_route": score_route,
                "score_penalty": score_penalty,
                "success_flag": success_flag,
                "duration_game": duration_game,
                "duration_system": duration_system,
                "ego_runs": aggregated["ego_runs"],
                "infractions": infractions or {},
                "agent_count": agent_count,
                "hug_score": aggregated.get("hug_score"),
                "hug_route_completion": aggregated.get("hug_route_completion"),
                "hug_epdm": aggregated.get("hug_epdm"),
                "negotiations_count": negotiation_count,
                "negotiation_rounds": negotiation_rounds,
                "negotiation_avg_rounds": negotiation_avg_rounds,
                "negotiation_median_rounds": negotiation_median_rounds,
                "negotiation_max_rounds": negotiation_max_rounds,
                "negotiation_suggestions": negotiation_suggestions,
                "negotiation_speakers": negotiation_speakers,
                "negotiation_cons_scores": negotiation_cons_scores,
                "negotiation_safety_scores": negotiation_safety_scores,
                "negotiation_efficiency_scores": negotiation_efficiency_scores,
                "negotiation_total_scores": negotiation_total_scores,
                "negotiation_min_distances": negotiation_min_distances,
                "negotiation_avg_consensus": negotiation_avg_consensus,
                "negotiation_avg_safety": negotiation_avg_safety,
                "negotiation_avg_efficiency": negotiation_avg_efficiency,
                "negotiation_avg_total": negotiation_avg_total,
                "negotiation_avg_min_distance": negotiation_avg_min_distance,
            }
        )

    if not route_entries:
        return None

    npc_entries = [entry for entry in route_entries if entry["mode"] == "NPC"]
    no_npc_entries = [entry for entry in route_entries if entry["mode"] == "No NPC"]

    hug_vals = [e["hug_score"] for e in route_entries if e.get("hug_score") is not None]
    hug_rc_vals = [e["hug_route_completion"] for e in route_entries if e.get("hug_route_completion") is not None]
    hug_epdm_vals_all = [e["hug_epdm"] for e in route_entries if e.get("hug_epdm") is not None]

    stats = {
        "total_route_entries": len(route_entries),
        "unique_routes": len({entry["route_name"] for entry in route_entries}),
        "npc_route_entries": len(npc_entries),
        "no_npc_route_entries": len(no_npc_entries),
        "with_infractions": len(infractions_summary),
        "total_ego_runs": total_ego_runs,
        "avg_hug": average(hug_vals) if hug_vals else None,
        "avg_hug_rc": average(hug_rc_vals) if hug_rc_vals else None,
        "avg_hug_epdm": average(hug_epdm_vals_all) if hug_epdm_vals_all else None,
        "hug_available": len(hug_vals) > 0,
        "avg_ds": average([entry["score_composed"] for entry in route_entries]),
        "avg_success": average([entry["success_flag"] for entry in route_entries]),
        "total_negotiations": total_negotiations,
        "routes_with_negotiations": sum(1 for entry in route_entries if entry["negotiations_count"] > 0),
        "avg_rounds_per_negotiation": (sum(negotiation_rounds_all) / total_negotiations) if total_negotiations else 0.0,
        "median_rounds_per_negotiation": statistics.median(negotiation_rounds_all) if negotiation_rounds_all else 0.0,
        "max_rounds_per_negotiation": max(negotiation_rounds_all) if negotiation_rounds_all else 0,
        "negotiation_suggestions": negotiation_suggestions_total,
        "avg_consensus": average(negotiation_cons_scores_all),
        "avg_safety": average(negotiation_safety_scores_all),
        "avg_efficiency": average(negotiation_efficiency_scores_all),
        "avg_negotiation_score": average(negotiation_total_scores_all),
        "avg_negotiation_min_distance": average(negotiation_min_distances_all),
    }

    communication_summary: Dict[str, Dict] = {
        "overall": {
            "routes": stats["total_route_entries"],
            "negotiations": stats["total_negotiations"],
            "routes_with_negotiations": stats["routes_with_negotiations"],
            "rounds": sum(negotiation_rounds_all),
            "rounds_list": negotiation_rounds_all,
            "avg_rounds": stats["avg_rounds_per_negotiation"],
            "median_rounds": stats["median_rounds_per_negotiation"],
            "max_rounds": stats["max_rounds_per_negotiation"],
            "suggestions": negotiation_suggestions_total,
            "cons_scores": negotiation_cons_scores_all,
            "avg_consensus": stats["avg_consensus"],
            "safety_scores": negotiation_safety_scores_all,
            "avg_safety": stats["avg_safety"],
            "efficiency_scores": negotiation_efficiency_scores_all,
            "avg_efficiency": stats["avg_efficiency"],
            "total_scores": negotiation_total_scores_all,
            "avg_total_score": stats["avg_negotiation_score"],
            "min_distances": negotiation_min_distances_all,
            "avg_min_distance": stats["avg_negotiation_min_distance"],
        },
        "by_mode": {},
    }

    mode_summaries: Dict[str, Dict] = {}
    for entry in route_entries:
        mode = entry["mode"]
        summary = mode_summaries.setdefault(
            mode,
            {
                "routes": 0,
                "negotiations": 0,
                "routes_with_negotiations": 0,
                "rounds_list": [],
                "suggestions": 0,
                "cons_scores": [],
                "safety_scores": [],
                "efficiency_scores": [],
                "total_scores": [],
                "min_distances": [],
            },
        )
        summary["routes"] += 1
        if entry["negotiations_count"] > 0:
            summary["negotiations"] += entry["negotiations_count"]
            summary["routes_with_negotiations"] += 1
            summary["rounds_list"].extend(entry["negotiation_rounds"])
            summary["suggestions"] += entry["negotiation_suggestions"]
            summary["cons_scores"].extend(entry["negotiation_cons_scores"])
            summary["safety_scores"].extend(entry["negotiation_safety_scores"])
            summary["efficiency_scores"].extend(entry["negotiation_efficiency_scores"])
            summary["total_scores"].extend(entry["negotiation_total_scores"])
            summary["min_distances"].extend(entry["negotiation_min_distances"])

    for mode, info in mode_summaries.items():
        rounds_list = info["rounds_list"]
        total_mode_negs = info["negotiations"]
        communication_summary["by_mode"][mode] = {
            "routes": info["routes"],
            "negotiations": total_mode_negs,
            "routes_with_negotiations": info["routes_with_negotiations"],
            "rounds": sum(rounds_list),
            "rounds_list": rounds_list,
            "avg_rounds": (sum(rounds_list) / total_mode_negs) if total_mode_negs else 0.0,
            "median_rounds": statistics.median(rounds_list) if rounds_list else 0.0,
            "max_rounds": max(rounds_list) if rounds_list else 0,
            "suggestions": info["suggestions"],
            "cons_scores": info["cons_scores"],
            "avg_consensus": average(info["cons_scores"]),
            "safety_scores": info["safety_scores"],
            "avg_safety": average(info["safety_scores"]),
            "efficiency_scores": info["efficiency_scores"],
            "avg_efficiency": average(info["efficiency_scores"]),
            "total_scores": info["total_scores"],
            "avg_total_score": average(info["total_scores"]),
            "min_distances": info["min_distances"],
            "avg_min_distance": average(info["min_distances"]),
        }

    return {
        "route_entries": route_entries,
        "infractions": infractions_summary,
        "stats": stats,
        "infraction_totals": dict(sorted(infraction_totals.items(), key=lambda item: item[0])),
        "communication_summary": communication_summary,
        "negotiation_records": negotiation_records,
        "negotiation_sources": negotiation_sources,
    }


def _fmt_hug(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else ""


def route_entries_for_table(entries: List[Dict]) -> List[List]:
    rows = []
    for entry in sorted(entries, key=route_entry_sort_key):
        rows.append(
            [
                entry["route_name"],
                entry["mode"],
                entry.get("agent_count", 0),
                entry.get("hug_score"),
                entry.get("hug_route_completion"),
                entry.get("hug_epdm"),
                entry["score_composed"],
                entry["score_route"],
                entry["score_penalty"],
                entry["success_flag"],
                entry["duration_game"],
                entry["duration_system"],
                entry["ego_runs"],
                infractions_to_string(entry.get("infractions", {})),
                entry.get("negotiations_count", 0),
                entry.get("negotiation_avg_rounds", 0.0),
                entry.get("negotiation_median_rounds", 0.0),
                entry.get("negotiation_max_rounds", 0),
                entry.get("negotiation_suggestions", 0),
                entry.get("negotiation_speakers", 0),
                entry.get("negotiation_avg_consensus", 0.0),
                entry.get("negotiation_avg_safety", 0.0),
                entry.get("negotiation_avg_efficiency", 0.0),
                entry.get("negotiation_avg_total", 0.0),
                entry.get("negotiation_avg_min_distance", 0.0),
            ]
        )
    return rows


def compute_route_pairs(entries: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    pairs = defaultdict(dict)
    for entry in entries:
        pairs[entry["route_name"]][entry["mode"]] = entry
    return pairs


def infractions_to_string(infractions: Dict[str, int]) -> str:
    if not infractions:
        return ""
    return ", ".join(f"{k}: {v}" for k, v in sorted(infractions.items()))



def slugify(value: str) -> str:
    cleaned = ''.join(char if char.isalnum() or char in ('-', '_') else '_' for char in value)
    cleaned = cleaned.strip('_')
    return cleaned or 'experiment'


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)



def render_console_report(label: str, result: Dict):
    stats = result["stats"]
    route_entries = result["route_entries"]
    infractions = result["infractions"]
    comm_summary = result.get("communication_summary", {})

    print(f"\n===== {label} =====")
    print(
        textwrap.fill(
            (
                f"Routes evaluated: {stats['total_route_entries']} "
                f"(No NPC: {stats['no_npc_route_entries']}, NPC: {stats['npc_route_entries']}). "
                f"Unique scenarios: {stats['unique_routes']}. "
                f"Aggregated ego runs: {stats['total_ego_runs']}."
            ),
            width=100,
        )
    )
    score_parts = []
    if stats.get("hug_available"):
        score_parts.append(f"Average HUG: {round(stats['avg_hug'], 4)}")
        if stats.get("avg_hug_rc") is not None:
            score_parts.append(f"HUG RC: {round(stats['avg_hug_rc'], 4)}")
        if stats.get("avg_hug_epdm") is not None:
            score_parts.append(f"EPDM: {round(stats['avg_hug_epdm'], 4)}")
    score_parts.append(f"Average DS: {round(stats['avg_ds'], 2)}")
    score_parts.append(f"Success rate: {round(stats['avg_success'] * 100, 1)}%")
    print(textwrap.fill(" · ".join(score_parts), width=100))

    hug_available = stats.get("hug_available", False)
    sort_key = (lambda e: e["hug_score"] if e.get("hug_score") is not None else -1) if hug_available else (lambda e: e["score_composed"])
    top_routes = sorted(route_entries, key=sort_key, reverse=True)[:5]
    if top_routes:
        if hug_available:
            table = [
                [
                    entry["route_name"],
                    entry["mode"],
                    f"{entry['hug_score']:.4f}" if entry.get("hug_score") is not None else "-",
                    f"{entry['hug_route_completion']:.4f}" if entry.get("hug_route_completion") is not None else "-",
                    f"{entry['hug_epdm']:.4f}" if entry.get("hug_epdm") is not None else "-",
                    round(entry["score_composed"], 2),
                    f"{round(entry['success_flag'] * 100, 1)}%",
                ]
                for entry in top_routes
            ]
            print("\nTop 5 Routes by HUG Score")
            print(tabulate(table, headers=["Route", "Mode", "HUG", "HUG RC", "EPDM", "DS", "Success"], tablefmt="github"))
        else:
            table = [
                [
                    entry["route_name"],
                    entry["mode"],
                    round(entry["score_composed"], 2),
                    round(entry["score_route"], 2),
                    round(entry["score_penalty"], 3),
                    f"{round(entry['success_flag'] * 100, 1)}%",
                ]
                for entry in top_routes
            ]
            print("\nTop 5 Routes by Driving Score")
            print(tabulate(table, headers=["Route", "Mode", "DS", "RC", "IS", "Success"], tablefmt="github"))

    route_pairs = compute_route_pairs(route_entries)
    penalty_rows = []
    for route_name, modes in route_pairs.items():
        npc_entry = modes.get("NPC")
        no_npc_entry = modes.get("No NPC")
        if not npc_entry or not no_npc_entry:
            continue
        delta = no_npc_entry["score_composed"] - npc_entry["score_composed"]
        if delta > 0.1:
            penalty_rows.append(
                [
                    route_name,
                    round(no_npc_entry["score_composed"], 2),
                    round(npc_entry["score_composed"], 2),
                    round(delta, 2),
                ]
            )

    if penalty_rows:
        penalty_rows.sort(key=lambda row: row[3], reverse=True)
        print("\nRoutes most impacted by NPC traffic")
        print(tabulate(penalty_rows[:5], headers=["Route", "No NPC DS", "NPC DS", "Delta DS"], tablefmt="github"))

    agent_summary = defaultdict(lambda: {"routes": 0, "modes": defaultdict(int), "ds": [], "negs": 0})
    for entry in route_entries:
        count = entry.get("agent_count", 0)
        summary = agent_summary[count]
        summary["routes"] += 1
        summary["modes"][entry.get("mode", "")] += 1
        summary["ds"].append(entry.get("score_composed", 0.0))
        summary["negs"] += entry.get("negotiations_count", 0)

    if agent_summary:
        rows = []
        for count in sorted(agent_summary.keys()):
            summary = agent_summary[count]
            routes = summary["routes"]
            avg_ds = average(summary["ds"])
            avg_negs = summary["negs"] / routes if routes else 0.0
            rows.append(
                [
                    count,
                    routes,
                    summary["modes"].get("No NPC", 0),
                    summary["modes"].get("NPC", 0),
                    round(avg_ds, 2),
                    round(avg_negs, 2),
                ]
            )
        print("\nAgent count summary")
        print(tabulate(rows, headers=["Agents", "Routes", "No NPC", "NPC", "Avg DS", "Avg Negotiations"], tablefmt="github"))

    overall_comm = comm_summary.get("overall", {})
    total_negotiations = overall_comm.get("negotiations", 0)
    if total_negotiations:
        avg_rounds = stats.get("avg_rounds_per_negotiation", 0.0)
        median_rounds = stats.get("median_rounds_per_negotiation", 0.0)
        avg_consensus = stats.get("avg_consensus", 0.0)
        avg_total = stats.get("avg_negotiation_score", 0.0)
        print(
            f"\nNegotiations: {total_negotiations} total "
            f"(routes with negotiations: {stats.get('routes_with_negotiations', 0)}). "
            f"Average rounds: {round(avg_rounds, 2)}, median rounds: {round(median_rounds, 2)}. "
            f"Avg consensus score: {round(avg_consensus, 2)}, avg negotiation total: {round(avg_total, 2)}."
        )
        mode_rows = []
        for mode, summary in comm_summary.get("by_mode", {}).items():
            mode_rows.append(
                [
                    mode,
                    summary["routes"],
                    summary["negotiations"],
                    summary["routes_with_negotiations"],
                    round(summary.get("avg_rounds", 0.0), 2),
                    round(summary.get("median_rounds", 0.0), 2),
                    summary.get("suggestions", 0),
                    round(summary.get("avg_consensus", 0.0), 2),
                    round(summary.get("avg_efficiency", 0.0), 2),
                    round(summary.get("avg_total_score", 0.0), 2),
                ]
            )
        if mode_rows:
            print(
                tabulate(
                    mode_rows,
                    headers=[
                        "Mode",
                        "Routes",
                        "Negotiations",
                        "Routes w/ Neg",
                        "Avg Rounds",
                        "Median",
                        "Suggestions",
                        "Avg Cons",
                        "Avg Eff",
                        "Avg Total",
                    ],
                    tablefmt="github",
                )
            )

    if infractions:
        print("\nRoutes with Infractions")
        inf_rows = [
            [item["route_name"], item["mode"], infractions_to_string(item["details"])]
            for item in infractions
        ]
        print(tabulate(inf_rows, headers=["Route", "Mode", "Details"], tablefmt="github"))
    else:
        print("\nNo infractions detected ✅")


def generate_npc_gap_chart(figures_dir: Path, route_pairs: Dict[str, Dict[str, Dict]], experiment_dir: Path):
    rows = []
    for route_name, modes in route_pairs.items():
        npc_entry = modes.get("NPC")
        no_npc_entry = modes.get("No NPC")
        if not npc_entry or not no_npc_entry:
            continue
        diff = no_npc_entry["score_composed"] - npc_entry["score_composed"]
        rows.append((route_name, no_npc_entry, npc_entry, diff))

    if not rows:
        return None

    rows.sort(key=lambda item: abs(item[3]), reverse=True)
    rows = rows[:15]

    fig, ax = plt.subplots(figsize=(11, 0.8 + 0.45 * len(rows)))
    y_positions = np.arange(len(rows))
    diffs = np.array([item[3] for item in rows])
    colors = np.where(diffs >= 0, "#d62728", "#2ca02c")
    ax.barh(y_positions, diffs, color=colors, height=0.6)

    ax.text(
        0.98,
        1.02,
        "Δ (No NPC - NPC)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#222222",
    )

    for pos, diff, row in zip(y_positions, diffs, rows):
        npc_score = row[2]["score_composed"]
        no_npc_score = row[1]["score_composed"]
        offset = 2 if diff >= 0 else -2
        align = "left" if diff >= 0 else "right"
        ax.text(diff + offset, pos + 0.15, f"{diff:+.1f}", va="center", ha=align, fontsize=9, color="#222222")
        ax.text(
            diff + offset,
            pos - 0.25,
            f"No NPC: {no_npc_score:.1f} | NPC: {npc_score:.1f}",
            va="center",
            ha=align,
            fontsize=8,
            color="#555555",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Driving Score difference (No NPC - NPC)")
    ax.set_title("Routes Most Impacted by NPC Traffic")
    ax.axvline(0, color="#444444", linewidth=1)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "npc_gap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_score_distribution(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    data = defaultdict(list)
    for entry in route_entries:
        data[entry["mode"]].append(entry["score_composed"])

    available_modes = [mode for mode in ("No NPC", "NPC") if data.get(mode)]
    if not available_modes:
        available_modes = sorted(data.keys(), key=lambda label: MODE_ORDER.get(label, len(MODE_ORDER)))
    if not available_modes:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = [data[mode] for mode in available_modes]
    box = ax.boxplot(box_data, patch_artist=True)
    palette = {"No NPC": "#1f77b4", "NPC": "#d62728"}
    for patch, mode in zip(box["boxes"], available_modes):
        patch.set_facecolor(palette.get(mode, "#7f7f7f"))
        patch.set_alpha(0.6)
    for whisker in box["whiskers"]:
        whisker.set_color("#444444")
    for cap in box["caps"]:
        cap.set_color("#444444")
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(2)

    ax.set_xticks(np.arange(1, len(available_modes) + 1))
    ax.set_xticklabels(available_modes)
    ax.set_ylabel("Driving Score (DS)")
    ax.set_title("Driving Score Distribution by Traffic Setting")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "score_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()

def generate_negotiation_volume_chart(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    rows = [
        (entry["route_name"], entry["mode"], entry.get("negotiations_count", 0))
        for entry in route_entries
        if entry.get("negotiations_count", 0) > 0
    ]
    if not rows:
        return None

    rows.sort(key=lambda item: item[2], reverse=True)
    rows = rows[:15]
    y_positions = np.arange(len(rows))
    counts = np.array([row[2] for row in rows])
    palette = {"No NPC": "#1f77b4", "NPC": "#ff7f0e"}
    colors = [palette.get(row[1], "#7f7f7f") for row in rows]

    fig, ax = plt.subplots(figsize=(11, 0.8 + 0.45 * len(rows)))
    ax.barh(y_positions, counts, color=colors, height=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row[0]} ({row[1]})" for row in rows])
    ax.set_xlabel("Negotiations Started")
    ax.set_title("Top Routes by Negotiation Count")
    for y, count in zip(y_positions, counts):
        ax.text(count + 0.1, y, str(int(count)), va="center", fontsize=9)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    handles = [Patch(facecolor=color, label=mode) for mode, color in palette.items()]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "negotiations_per_route.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_negotiation_rounds_chart(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    rounds = [rounds for entry in route_entries for rounds in entry.get("negotiation_rounds", [])]
    if not rounds:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = range(1, max(rounds) + 2)
    ax.hist(rounds, bins=bins, color="#9467bd", edgecolor="#ffffff", alpha=0.85, align="left")
    ax.set_xlabel("Rounds per Negotiation")
    ax.set_ylabel("Number of Negotiations")
    ax.set_title("Negotiation Round Distribution")
    ax.set_xticks(range(1, max(rounds) + 1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "negotiation_rounds_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_agent_count_distribution(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    counts = defaultdict(lambda: defaultdict(int))
    modes_present = set()
    for entry in route_entries:
        agent_count = entry.get("agent_count", 0)
        mode = entry.get("mode") or DEFAULT_MODE_LABEL
        counts[agent_count][mode] += 1
        modes_present.add(mode)

    if not counts:
        return None

    sorted_counts = sorted(counts.keys())
    if not modes_present:
        modes_present.add(DEFAULT_MODE_LABEL)
    modes = sorted(modes_present, key=lambda label: MODE_ORDER.get(label, len(MODE_ORDER)))
    x = np.arange(len(sorted_counts))
    width = 0.8 / max(len(modes), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, mode in enumerate(modes):
        values = [counts[count].get(mode, 0) for count in sorted_counts]
        offset = (idx - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels([str(count) for count in sorted_counts])
    ax.set_xlabel("Agent count")
    ax.set_ylabel("Number of routes")
    ax.set_title("Routes by Agent Count and Traffic Setting")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "agent_count_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_agent_ds_boxplot(figures_dir: Path, route_entries: List[Dict], experiment_dir: Path):
    groups = defaultdict(list)
    for entry in route_entries:
        agent_count = entry.get("agent_count")
        score = entry.get("score_composed")
        if agent_count is None or score is None:
            continue
        groups[agent_count].append(score)

    if not groups:
        return None

    sorted_counts = sorted(groups.keys())
    data = [groups[count] for count in sorted_counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, patch_artist=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(sorted_counts)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in box["medians"]:
        median.set_color("#2f2f2f")
        median.set_linewidth(2)

    ax.set_xticks(np.arange(1, len(sorted_counts) + 1))
    ax.set_xticklabels([str(count) for count in sorted_counts])
    ax.set_xlabel("Agent count")
    ax.set_ylabel("Driving Score (DS)")
    ax.set_title("Driving Score Distribution by Agent Count")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "agent_ds_boxplot.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()


def generate_infractions_chart(figures_dir: Path, infractions: List[Dict], experiment_dir: Path):
    rows = []
    for item in infractions:
        total = sum(item["details"].values())
        if total <= 0:
            continue
        rows.append((item["route_name"], item["mode"], item["details"], total))

    if not rows:
        return None

    rows.sort(key=lambda item: item[3], reverse=True)
    rows = rows[:12]

    infraction_types = sorted({key for _, _, details, _ in rows for key in details})
    if not infraction_types:
        return None

    fig, ax = plt.subplots(figsize=(10, 0.8 + 0.45 * len(rows)))
    y_positions = np.arange(len(rows))
    cumulative = np.zeros(len(rows))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(infraction_types), 1)) for i in range(len(infraction_types))]

    for idx, inf_type in enumerate(infraction_types):
        segment = np.array([details.get(inf_type, 0) for _, _, details, _ in rows])
        if not segment.any():
            continue
        ax.barh(
            y_positions,
            segment,
            left=cumulative,
            height=0.55,
            color=colors[idx],
            label=inf_type,
        )
        cumulative += segment

    for pos, (_, mode, _, total) in enumerate(rows):
        ax.text(total + 0.1, pos, f"{int(total)}", va="center", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{route} ({mode})" for route, mode, _, _ in rows])
    ax.set_xlabel("Infraction events")
    ax.set_title("Most Frequent Infractions")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02))
    fig.tight_layout()

    ensure_directory(figures_dir)
    path = figures_dir / "infractions_breakdown.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(experiment_dir).as_posix()

def write_csv_file(path: Path, headers: List[str], rows: List[List]):
    ensure_directory(path.parent)
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def generate_figures(experiment_dir: Path, result: Dict):
    figures_dir = experiment_dir / "figures"
    ensure_directory(figures_dir)

    figure_paths = {}

    npc_gap_path = generate_npc_gap_chart(figures_dir, compute_route_pairs(result["route_entries"]), experiment_dir)
    if npc_gap_path:
        figure_paths["npc_gap.png"] = npc_gap_path

    score_dist_path = generate_score_distribution(figures_dir, result["route_entries"], experiment_dir)
    if score_dist_path:
        figure_paths["score_distribution.png"] = score_dist_path

    neg_vol_path = generate_negotiation_volume_chart(figures_dir, result["route_entries"], experiment_dir)
    if neg_vol_path:
        figure_paths["negotiations_per_route.png"] = neg_vol_path

    neg_rounds_path = generate_negotiation_rounds_chart(figures_dir, result["route_entries"], experiment_dir)
    if neg_rounds_path:
        figure_paths["negotiation_rounds_distribution.png"] = neg_rounds_path

    agent_dist_path = generate_agent_count_distribution(figures_dir, result["route_entries"], experiment_dir)
    if agent_dist_path:
        figure_paths["agent_count_distribution.png"] = agent_dist_path

    agent_ds_path = generate_agent_ds_boxplot(figures_dir, result["route_entries"], experiment_dir)
    if agent_ds_path:
        figure_paths["agent_ds_boxplot.png"] = agent_ds_path

    infractions_path = generate_infractions_chart(figures_dir, result["infractions"], experiment_dir)
    if infractions_path:
        figure_paths["infractions_breakdown.png"] = infractions_path

    return figure_paths

def create_markdown_report(label: str, result: Dict, resources: Optional[Dict] = None) -> str:
    stats = result["stats"]
    route_entries = result["route_entries"]
    category_summaries = result["category_summaries"]
    infractions = result["infractions"]

    per_route_csv = None
    infractions_csv = None
    negotiation_summary_csv = None
    negotiation_detailed_csv = None
    negotiation_report = None
    negotiation_tables: Dict[str, str] = {}
    negotiation_figures_extra: Dict[str, str] = {}
    negotiation_logs_dir = None
    figures: Dict[str, str] = {}

    if resources:
        per_route_csv = resources.get("per_route_csv")
        infractions_csv = resources.get("infractions_csv")
        negotiation_summary_csv = resources.get("negotiation_summary_csv")
        negotiation_detailed_csv = resources.get("negotiation_detailed_csv")
        negotiation_report = resources.get("negotiation_report")
        negotiation_tables = resources.get("negotiation_tables", {})
        negotiation_figures_extra = resources.get("negotiation_figures", {})
        negotiation_logs_dir = resources.get("negotiation_logs_dir")
        figures = resources.get("figures", {})

    hug_available = stats.get("hug_available", False)

    lines: List[str] = []
    lines.append(f"# Experiment Report · {label}")
    lines.append("")
    lines.append("## Overview")
    overview_items = [
        f"- Total route evaluations: **{stats['total_route_entries']}** (No NPC: {stats['no_npc_route_entries']}, NPC: {stats['npc_route_entries']})",
        f"- Unique route scenarios: **{stats['unique_routes']}**",
        f"- Aggregated ego runs: **{stats['total_ego_runs']}**",
    ]
    if hug_available:
        overview_items.append(f"- Average HUG score: **{round(stats['avg_hug'], 4)}**")
        if stats.get("avg_hug_rc") is not None:
            overview_items.append(f"- Average HUG route completion: **{round(stats['avg_hug_rc'], 4)}**")
        if stats.get("avg_hug_epdm") is not None:
            overview_items.append(f"- Average EPDM: **{round(stats['avg_hug_epdm'], 4)}**")
    overview_items.extend([
        f"- Average driving score (DS): **{round(stats['avg_ds'], 2)}**",
        f"- Average success rate: **{round(stats['avg_success'] * 100, 1)}%**",
    ])
    lines.extend(overview_items)
    lines.append("")

    if figures:
        lines.append("## Visual Highlights")
        ordered_figures = [
            ("npc_gap.png", "Routes where NPC traffic changes the driving score (positive values mean NPCs make the scenario harder)."),
            ("negotiations_per_route.png", "Top routes by number of negotiations started."),
            ("negotiation_rounds_distribution.png", "Distribution of negotiation lengths measured in rounds."),
            ("negotiations_avg_rounds_per_scenario.png", "Average rounds per negotiation for each scenario with descriptive labels."),
            ("negotiations_rounds_by_agent_count.png", "Negotiation round distribution across agent counts."),
            ("negotiations_rounds_by_setting.png", "Negotiation rounds grouped by scenario setting."),
            ("negotiations_rounds_vs_agent_count.png", "Average negotiation rounds versus agent count."),
            ("agent_count_distribution.png", "Routes per agent count split by traffic setting."),
            ("agent_ds_boxplot.png", "Driving score distribution grouped by agent count."),
            ("score_distribution.png", "Distribution of driving scores for NPC and no-NPC runs."),
            ("infractions_breakdown.png", "Stacked view of infractions per route; repeated colors show multiple events of the same type."),
        ]
        consumed = set()
        for filename, caption in ordered_figures:
            path = figures.get(filename)
            if not path:
                continue
            lines.append(f"![{caption}]({path})")
            lines.append("")
            consumed.add(filename)
        for filename, path in figures.items():
            if filename in consumed:
                continue
            lines.append(f"![{filename}]({path})")
            lines.append("")

    lines.append("## Per-Route Summary")
    if per_route_csv:
        lines.append(f"[Download full per-route dataset]({per_route_csv})")

    sort_key_md = (lambda e: e["hug_score"] if e.get("hug_score") is not None else -1) if hug_available else (lambda e: e["score_composed"])
    top_routes = sorted(route_entries, key=sort_key_md, reverse=True)[:15]
    if top_routes:
        if hug_available:
            table = [
                [
                    entry["route_name"],
                    entry["mode"],
                    f"{entry['hug_score']:.4f}" if entry.get("hug_score") is not None else "-",
                    f"{entry['hug_route_completion']:.4f}" if entry.get("hug_route_completion") is not None else "-",
                    f"{entry['hug_epdm']:.4f}" if entry.get("hug_epdm") is not None else "-",
                    round(entry["score_composed"], 2),
                    f"{round(entry['success_flag'] * 100, 1)}%",
                    round(entry["duration_game"], 2),
                ]
                for entry in top_routes
            ]
            lines.append("")
            lines.append("Top 15 routes by HUG score:")
            lines.append(tabulate(table, headers=["Route", "Mode", "HUG", "HUG RC", "EPDM", "DS", "Success", "Game Time (s)"], tablefmt="github"))
        else:
            table = [
                [
                    entry["route_name"],
                    entry["mode"],
                    round(entry["score_composed"], 2),
                    round(entry["score_route"], 2),
                    round(entry["score_penalty"], 3),
                    f"{round(entry['success_flag'] * 100, 1)}%",
                    round(entry["duration_game"], 2),
                ]
                for entry in top_routes
            ]
            lines.append("")
            lines.append("Top 15 routes by driving score:")
            lines.append(tabulate(table, headers=["Route", "Mode", "DS", "RC", "IS", "Success", "Game Time (s)"], tablefmt="github"))
    else:
        lines.append("No route data available.")
    lines.append("")

    route_pairs = compute_route_pairs(route_entries)
    penalty_rows = []
    benefit_rows = []
    for route_name, modes in route_pairs.items():
        npc_entry = modes.get("NPC")
        no_npc_entry = modes.get("No NPC")
        if not npc_entry or not no_npc_entry:
            continue
        delta = no_npc_entry["score_composed"] - npc_entry["score_composed"]
        record = {
            "route": route_name,
            "no_npc": no_npc_entry["score_composed"],
            "npc": npc_entry["score_composed"],
            "delta": delta,
        }
        if delta > 0:
            penalty_rows.append(record)
        elif delta < 0:
            benefit_rows.append(record)

    penalty_rows.sort(key=lambda item: item["delta"], reverse=True)
    benefit_rows.sort(key=lambda item: item["delta"])

    if penalty_rows:
        lines.append("Routes most impacted by NPC traffic (positive delta = score drop with NPCs present):")
        penalty_table = [
            [row["route"], round(row["no_npc"], 2), round(row["npc"], 2), round(row["delta"], 2)]
            for row in penalty_rows[:10]
        ]
        lines.append(tabulate(penalty_table, headers=["Route", "No NPC DS", "NPC DS", "Delta DS"], tablefmt="github"))
        lines.append("")

    if benefit_rows:
        lines.append("Routes where NPC traffic improved the score (negative delta):")
        benefit_table = [
            [row["route"], round(row["no_npc"], 2), round(row["npc"], 2), round(row["delta"], 2)]
            for row in benefit_rows[:5]
        ]
        lines.append(tabulate(benefit_table, headers=["Route", "No NPC DS", "NPC DS", "Delta DS"], tablefmt="github"))
        lines.append("")

    lowest_routes = sorted(route_entries, key=lambda entry: entry["score_composed"])[:10]
    if lowest_routes:
        lines.append("Routes with the lowest driving scores:")
        lines.append(
            tabulate(
                [
                    [
                        entry["route_name"],
                        entry["mode"],
                        round(entry["score_composed"], 2),
                        f"{round(entry['success_flag'] * 100, 1)}%",
                        round(entry["duration_game"], 2),
                    ]
                    for entry in lowest_routes
                ],
                headers=["Route", "Mode", "DS", "Success", "Game Time (s)"],
                tablefmt="github",
            )
        )
        lines.append("")

    agent_summary = defaultdict(lambda: {"routes": 0, "modes": defaultdict(int), "ds": [], "negs": 0})
    for entry in route_entries:
        count = entry.get("agent_count", 0)
        summary = agent_summary[count]
        summary["routes"] += 1
        summary["modes"][entry.get("mode", "")] += 1
        summary["ds"].append(entry.get("score_composed", 0.0))
        summary["negs"] += entry.get("negotiations_count", 0)

    if agent_summary:
        lines.append("## Agent Composition")
        rows = []
        for count in sorted(agent_summary.keys()):
            summary = agent_summary[count]
            routes = summary["routes"]
            avg_ds = average(summary["ds"])
            avg_negs = summary["negs"] / routes if routes else 0.0
            rows.append(
                [
                    count,
                    routes,
                    summary["modes"].get("No NPC", 0),
                    summary["modes"].get("NPC", 0),
                    round(avg_ds, 2),
                    round(avg_negs, 2),
                ]
            )
        lines.append(
            tabulate(
                rows,
                headers=["Agents", "Routes", "No NPC", "NPC", "Avg DS", "Avg Negotiations"],
                tablefmt="github",
            )
        )
        lines.append("")

    comm_summary = result.get("communication_summary", {})
    overall_comm = comm_summary.get("overall", {})
    lines.append("## Communication Analysis")
    total_negs = overall_comm.get("negotiations", 0)
    if total_negs:
        avg_rounds = round(stats.get("avg_rounds_per_negotiation", 0.0), 2)
        median_rounds = round(stats.get("median_rounds_per_negotiation", 0.0), 2)
        avg_cons = round(stats.get("avg_consensus", 0.0), 2)
        avg_total = round(stats.get("avg_negotiation_score", 0.0), 2)
        avg_min = round(stats.get("avg_negotiation_min_distance", 0.0), 2)
        lines.append(
            f"- Negotiations started: **{total_negs}** across {overall_comm.get('routes_with_negotiations', 0)} routes"
        )
        lines.append(f"- Average rounds per negotiation: **{avg_rounds}** (median {median_rounds})")
        lines.append(f"- Negotiations containing suggestions: **{overall_comm.get('suggestions', 0)}**")
        lines.append(
            f"- Average consensus score: **{avg_cons}**, average negotiation total: **{avg_total}**, average min distance: **{avg_min}**"
        )
        lines.append("")
        mode_rows = []
        for mode, summary in comm_summary.get("by_mode", {}).items():
            mode_rows.append(
                [
                    mode,
                    summary["routes"],
                    summary["negotiations"],
                    summary["routes_with_negotiations"],
                    round(summary.get("avg_rounds", 0.0), 2),
                    round(summary.get("median_rounds", 0.0), 2),
                    summary.get("suggestions", 0),
                    round(summary.get("avg_consensus", 0.0), 2),
                    round(summary.get("avg_efficiency", 0.0), 2),
                    round(summary.get("avg_total_score", 0.0), 2),
                ]
            )
        if mode_rows:
            lines.append(
                tabulate(
                    mode_rows,
                    headers=[
                        "Mode",
                        "Routes",
                        "Negotiations",
                        "Routes w/ Neg",
                        "Avg Rounds",
                        "Median",
                        "Suggestions",
                        "Avg Cons",
                        "Avg Eff",
                        "Avg Total",
                    ],
                    tablefmt="github",
                )
            )
            lines.append("")
        if negotiation_summary_csv:
            lines.append(f"[Download negotiation summary]({negotiation_summary_csv})")
        if negotiation_detailed_csv:
            lines.append(f"[Download negotiation detail]({negotiation_detailed_csv})")
        if negotiation_report:
            lines.append(f"[Download negotiation text report]({negotiation_report})")
        if negotiation_tables:
            table_labels = {
                "detailed": "Detailed records (full)",
                "scenario": "Per-scenario summary",
                "agent": "Agent-count breakdown",
                "base": "Base-type breakdown",
                "flavor": "Scenario-setting breakdown",
            }
            lines.append("Additional negotiation tables:")
            for key, path in sorted(negotiation_tables.items()):
                label = table_labels.get(key, key.replace("_", " ").title())
                lines.append(f"- {label}: [{path}]({path})")
        if negotiation_figures_extra:
            figure_labels = {
                "avg_rounds_per_scenario": "Average rounds per scenario",
                "rounds_by_agent_count": "Round distribution by agent count",
                "rounds_by_setting": "Round distribution by scenario setting",
                "rounds_vs_agent_count": "Average rounds versus agent count",
            }
            lines.append("Additional negotiation figures:")
            for key, path in sorted(negotiation_figures_extra.items()):
                label = figure_labels.get(key, key.replace("_", " ").title())
                lines.append(f"- {label}: [{path}]({path})")
        if negotiation_logs_dir:
            lines.append(f"Collected negotiation logs copied to `{negotiation_logs_dir}`")
        lines.append("")
    else:
        lines.append("No negotiations were recorded in these runs.")
        lines.append("")

    lines.append("## Infractions")
    if infractions:
        if infractions_csv:
            lines.append(f"[Download detailed infractions]({infractions_csv})")
        lines.append("")
        lines.append(
            "Each value represents the number of times that infraction occurred. "
            "Multiple counts for the same infraction indicate repeated events within the same route run."
        )
        lines.append("")
        totals = result.get("infraction_totals", {})
        if totals:
            lines.append("Total infraction events observed across all evaluations:")
            for key, count in sorted(totals.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"- **{key}**: {count}")
            lines.append("")
        for item in infractions:
            lines.append(
                f"- **{item['route_name']}** ({item['mode']}): {infractions_to_string(item['details'])}"
            )
    else:
        lines.append("No infractions recorded.")
    lines.append("")

    lines.append("")
    return "\n".join(lines).strip() + "\n"

def write_experiment_outputs(base_dir: Path, label: str, result: Dict, *, nest: bool):
    experiment_dir = base_dir / slugify(label) if nest else base_dir
    ensure_directory(experiment_dir)

    tables_dir = experiment_dir / "tables"
    ensure_directory(tables_dir)

    route_rows = route_entries_for_table(result["route_entries"])
    route_csv_path = tables_dir / "per_route_summary.csv"
    write_csv_file(
        route_csv_path,
        [
            "Route",
            "Mode",
            "Agent Count",
            "HUG Score",
            "HUG Route Completion",
            "HUG EPDM",
            "DS",
            "RC",
            "IS",
            "Success Rate",
            "Game Time (s)",
            "System Time (s)",
            "Ego Runs",
            "Infractions",
            "Negotiations",
            "Avg Rounds",
            "Median Rounds",
            "Max Rounds",
            "Suggestions",
            "Unique Speakers",
            "Avg Consensus",
            "Avg Safety",
            "Avg Efficiency",
            "Avg Total Score",
            "Avg Min Distance",
        ],
        [
            [
                row[0],
                row[1],
                str(int(row[2])) if isinstance(row[2], (int, float)) else row[2],
                f"{row[3]:.4f}" if row[3] is not None else "",
                f"{row[4]:.4f}" if row[4] is not None else "",
                f"{row[5]:.4f}" if row[5] is not None else "",
                f"{row[6]:.3f}",
                f"{row[7]:.3f}",
                f"{row[8]:.4f}",
                f"{row[9]:.3f}",
                f"{row[10]:.2f}",
                f"{row[11]:.2f}",
                str(int(row[12])) if isinstance(row[12], (int, float)) else row[12],
                row[13],
                str(int(row[14])) if isinstance(row[14], (int, float)) else row[14],
                f"{row[15]:.2f}",
                f"{row[16]:.2f}",
                str(int(row[17])) if isinstance(row[17], (int, float)) else row[17],
                str(int(row[18])) if isinstance(row[18], (int, float)) else row[18],
                str(int(row[19])) if isinstance(row[19], (int, float)) else row[19],
                f"{row[20]:.2f}",
                f"{row[21]:.2f}",
                f"{row[22]:.2f}",
                f"{row[23]:.2f}",
                f"{row[24]:.2f}",
            ]
            for row in route_rows
        ],
    )

    infractions_csv_path = None
    if result["infractions"]:
        infractions_rows = []
        for item in result["infractions"]:
            for key, count in item["details"].items():
                infractions_rows.append([item["route_name"], item["mode"], key, count])
        infractions_rows.sort(key=lambda row: (row[0], row[1], row[2]))
        infractions_csv = tables_dir / "infractions.csv"
        write_csv_file(
            infractions_csv,
            ["Route", "Mode", "Infraction", "Count"],
            infractions_rows,
        )
        infractions_csv_path = infractions_csv.relative_to(experiment_dir).as_posix()

    figure_paths = generate_figures(experiment_dir, result)

    negotiation_summary_csv_path = None
    negotiation_detailed_csv_path = None
    negotiation_report_path = None
    negotiation_extra_tables: Dict[str, str] = {}
    negotiation_extra_figures: Dict[str, str] = {}
    collected_logs_rel_path = None

    negotiation_records = result.get("negotiation_records") or []
    if negotiation_records:
        detailed_df = pd.DataFrame.from_records(negotiation_records)
        if not detailed_df.empty:
            scenario_agent_map = {
                entry["route_id"]: entry.get("agent_count")
                for entry in result["route_entries"]
            }

            def infer_mode(name: str) -> str:
                if isinstance(name, str):
                    if name.startswith("Interdrive_no_npc_"):
                        return "No NPC"
                    if name.startswith("Interdrive_npc_"):
                        return "NPC"
                return ""

            if "scenario" not in detailed_df.columns:
                detailed_df["scenario"] = detailed_df.get("route_id", "")
            if "mode" not in detailed_df.columns:
                detailed_df["mode"] = detailed_df["scenario"].map(infer_mode)
            else:
                detailed_df["mode"] = detailed_df["mode"].replace("", pd.NA)
                detailed_df["mode"] = detailed_df["mode"].fillna(detailed_df["scenario"].map(infer_mode))

            mapped_agents = detailed_df["scenario"].map(scenario_agent_map)
            if "agent_count" in detailed_df.columns:
                mapped_agents = mapped_agents.fillna(detailed_df["agent_count"])
            detailed_df["agent_count"] = mapped_agents

            for column in ("base_type", "sub_type"):
                if column not in detailed_df.columns:
                    detailed_df[column] = ""
                else:
                    detailed_df[column] = detailed_df[column].fillna("")

            detailed_df["scenario_type_label"] = detailed_df.apply(
                lambda row: scenario_type_label(row["base_type"], row["sub_type"]),
                axis=1,
            )
            detailed_df["scenario_bar_label"] = detailed_df.apply(
                lambda row: scenario_bar_label(
                    row.get("route_id", row["scenario"]),
                    row.get("agent_count"),
                    row["base_type"],
                    row["sub_type"],
                ),
                axis=1,
            )

            detailed_df["suggestion"] = detailed_df["suggestion"].astype(int)
            detailed_df["rounds"] = detailed_df["rounds"].astype(float)

            scenario_stats = negotiation_scenario_summary(detailed_df)
            agent_stats_df = negotiation_group_summary(detailed_df, "agent_count")
            base_stats_df = negotiation_group_summary(detailed_df, "base_type")
            flavor_stats_df = negotiation_flavor_summary(detailed_df)
            overall_neg_stats = negotiation_overall_statistics(detailed_df)

            negotiation_summary_csv = tables_dir / "negotiations_summary.csv"
            summary_columns = [
                ("scenario", "Scenario"),
                ("mode", "Mode"),
                ("agent_count", "Agent Count"),
                ("negotiations", "Negotiations"),
                ("mean_rounds", "Avg Rounds"),
                ("median_rounds", "Median Rounds"),
                ("max_rounds", "Max Rounds"),
                ("suggestions", "Suggestions"),
                ("avg_words", "Avg Words/Msg"),
                ("avg_chars", "Avg Chars/Msg"),
                ("total_rounds", "Total Rounds"),
                ("scenario_type_label", "Scenario Type"),
            ]
            summary_headers = dict(summary_columns)
            summary_keys = [col for col, _ in summary_columns]
            if scenario_stats.empty:
                summary_df = pd.DataFrame(columns=summary_headers.values())
            else:
                summary_df = scenario_stats[summary_keys].rename(columns=summary_headers)
                summary_df.sort_values(["Mode", "Scenario"], inplace=True, ignore_index=True)
            summary_df.to_csv(negotiation_summary_csv, index=False)
            negotiation_summary_csv_path = negotiation_summary_csv.relative_to(experiment_dir).as_posix()

            negotiation_detailed_csv = tables_dir / "negotiations_detailed.csv"
            detailed_export_columns = [
                ("scenario", "Scenario"),
                ("mode", "Mode"),
                ("run_id", "Run"),
                ("stage", "Stage"),
                ("event", "Event"),
                ("timestamp", "Timestamp"),
                ("rounds", "Rounds"),
                ("unique_speakers", "Unique Speakers"),
                ("avg_words_per_message", "Avg Words/Msg"),
                ("avg_chars_per_message", "Avg Chars/Msg"),
                ("suggestion", "Suggestion"),
                ("cons_score", "Consensus"),
                ("safety_score", "Safety"),
                ("efficiency_score", "Efficiency"),
                ("total_score", "Total Score"),
                ("min_distance", "Min Distance"),
                ("route_id", "Route ID"),
                ("agent_count", "Agent Count"),
                ("base_type", "Base Type"),
                ("sub_type", "Sub Type"),
            ]
            detailed_export = detailed_df[[col for col, _ in detailed_export_columns]].rename(
                columns=dict(detailed_export_columns)
            )
            detailed_export.to_csv(negotiation_detailed_csv, index=False)
            negotiation_detailed_csv_path = negotiation_detailed_csv.relative_to(experiment_dir).as_posix()

            extra_table_paths = save_negotiation_tables(
                detailed_df,
                scenario_stats,
                agent_stats_df,
                base_stats_df,
                flavor_stats_df,
                tables_dir,
            )
            negotiation_extra_tables = {
                key: path.relative_to(experiment_dir).as_posix() for key, path in extra_table_paths.items()
            }

            negotiation_figures_dir = experiment_dir / "figures" / "negotiations"
            extra_figure_paths = save_negotiation_figures(detailed_df, scenario_stats, negotiation_figures_dir)
            for path in extra_figure_paths.values():
                figure_paths[path.name] = path.relative_to(experiment_dir).as_posix()
            negotiation_extra_figures = {
                key: path.relative_to(experiment_dir).as_posix() for key, path in extra_figure_paths.items()
            }

            negotiation_report_path = tables_dir / "negotiations_report.txt"
            negotiation_report_text = render_negotiation_report(
                overall_neg_stats, agent_stats_df, base_stats_df, flavor_stats_df
            )
            negotiation_report_path.write_text(negotiation_report_text + "\n", encoding="utf-8")
            negotiation_report_path = negotiation_report_path.relative_to(experiment_dir).as_posix()

    negotiation_sources = result.get("negotiation_sources") or []
    if negotiation_sources:
        logs_dir = experiment_dir / "collected_nego_logs"
        ensure_directory(logs_dir)
        for source in negotiation_sources:
            target = logs_dir / f"{source.scenario_name}.json"
            try:
                shutil.copy2(source.source, target)
            except FileNotFoundError:
                continue
        collected_logs_rel_path = logs_dir.relative_to(experiment_dir).as_posix()

    resources = {
        "per_route_csv": route_csv_path.relative_to(experiment_dir).as_posix(),
        "infractions_csv": infractions_csv_path,
        "figures": figure_paths,
        "negotiation_summary_csv": negotiation_summary_csv_path,
        "negotiation_detailed_csv": negotiation_detailed_csv_path,
        "negotiation_report": negotiation_report_path,
        "negotiation_tables": negotiation_extra_tables,
        "negotiation_figures": negotiation_extra_figures,
        "negotiation_logs_dir": collected_logs_rel_path,
    }

    markdown_content = create_markdown_report(label, result, resources)
    markdown_path = experiment_dir / "report.md"
    markdown_path.write_text(markdown_content)

    return {
        "experiment_dir": experiment_dir,
        "markdown_path": markdown_path,
        "markdown_content": markdown_content,
        "resources": resources,
    }


def resolve_results_dir(path: str):
    if not os.path.isdir(path):
        return None
    v2x_path = os.path.join(path, "v2x_final")
    if os.path.isdir(v2x_path):
        return v2x_path
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CARLA leaderboard results for NPC/no-NPC experiments."
    )
    parser.add_argument(
        "base_path",
        help="Base directory containing results or a direct path to a v2x_final folder.",
    )
    parser.add_argument(
        "suffixes",
        nargs="*",
        help="Optional experiment suffixes to append to base_path (each will be analyzed).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        help="Directory where detailed reports (markdown, CSV, figures) will be written.",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        dest="markdown_path",
        help="If provided, aggregate markdown for all experiments into this single file.",
    )

    args = parser.parse_args()

    suffixes = args.suffixes if args.suffixes else [None]
    expected_multiple = len(suffixes) > 1
    any_valid = False
    markdown_sections: Optional[List[str]] = [] if args.markdown_path else None

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir:
        ensure_directory(output_dir)

    for suffix in suffixes:
        target_path = args.base_path if suffix is None else os.path.join(args.base_path, suffix)
        resolved = resolve_results_dir(target_path)
        if not resolved:
            print(f"Invalid path: {target_path}")
            continue

        label = (
            suffix
            if suffix is not None
            else os.path.basename(os.path.normpath(target_path))
        )
        result = analyze_results(resolved)
        if not result:
            print(f"No completed route results found in {resolved}")
            continue

        render_console_report(label, result)

        if output_dir:
            outputs = write_experiment_outputs(output_dir, label, result, nest=expected_multiple)
            if markdown_sections is not None:
                markdown_sections.append(outputs["markdown_content"])
        elif markdown_sections is not None:
            markdown_sections.append(create_markdown_report(label, result))

        any_valid = True

    if markdown_sections is not None and markdown_sections:
        with open(args.markdown_path, "w") as outfile:
            outfile.write("\n\n---\n\n".join(markdown_sections))
            outfile.write("\n")

    if not any_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
