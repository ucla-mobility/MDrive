#!/usr/bin/env python3
"""
Schema-based scenario generation script.

Generates JSON scenario specs and runs them through the pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scenario_generator import (
    SchemaGenerationLoop,
    SchemaGenerationLoopConfig,
    DEFAULT_SCHEMA_CATEGORIES,
    get_available_categories,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate JSON schema scenarios and run the pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_schema_scenarios.py --output-dir log_schema\n"
            "  python generate_schema_scenarios.py --categories \"Unprotected Left Turn\" --template-only\n"
            "  python generate_schema_scenarios.py --count-per-combination 2\n"
        ),
    )

    parser.add_argument(
        "--categories", nargs="+",
        help="Category names to generate (default: reduced test set)"
    )
    parser.add_argument(
        "--count-per-combination", type=int, default=1,
        help="Scenarios per category (default: 1)"
    )
    parser.add_argument(
        "--output-dir", default="log_schema",
        help="Output directory for results and pipeline artifacts"
    )
    parser.add_argument(
        "--town", default="Town05",
        help="CARLA town to use (default: Town05)"
    )
    parser.add_argument(
        "--highway-town", default="Town06",
        help="CARLA town to use for highway categories (default: Town06)"
    )
    parser.add_argument(
        "--t-junction-town", default="Town02",
        help="CARLA town to use for T-junction categories (default: Town02)"
    )
    parser.add_argument(
        "--two-lane-corridor-town", default="Town02",
        help="CARLA town to use for two-lane corridor categories (default: Town02)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-32B-Instruct-AWQ)"
    )
    parser.add_argument(
        "--template-only", action="store_true",
        help="Skip the LLM and use deterministic template specs"
    )
    parser.add_argument(
        "--no-template-fallback", action="store_true",
        help="Disable template fallback if LLM generation fails"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max attempts per scenario (default: 3)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum validation score to accept a scenario"
    )
    parser.add_argument(
        "--show-pipeline", action="store_true",
        help="Show pipeline output (not muted)"
    )
    parser.add_argument(
        "--all-prints", action="store_true",
        help="Enable verbose logging and show all pipeline output"
    )
    parser.add_argument(
        "--routes-out-dir", default="routes",
        help="Directory to write routes output (default: routes). Use --no-routes to disable."
    )
    parser.add_argument(
        "--no-routes", action="store_true",
        help="Disable route output entirely"
    )
    parser.add_argument(
        "--routes-ego-num", type=int, default=None,
        help="Override ego vehicle count for route conversion (default: auto-detect)"
    )
    parser.add_argument(
        "--carla-host", type=str, default="127.0.0.1",
        help="CARLA host for route alignment (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--carla-port", type=int, default=2000,
        help="CARLA port for route alignment (default: 2000)"
    )
    parser.add_argument(
        "--no-align-routes", action="store_true",
        help="Disable CARLA route alignment"
    )
    parser.add_argument(
        "--list-categories", action="store_true",
        help="List available categories and exit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    if args.list_categories:
        print("\nAvailable Categories:")
        print("=" * 40)
        for cat in get_available_categories():
            print(f"  - {cat}")
        print("\nDefault schema categories:")
        for cat in DEFAULT_SCHEMA_CATEGORIES:
            print(f"  - {cat}")
        return

    categories = args.categories or DEFAULT_SCHEMA_CATEGORIES
    available = set(get_available_categories())
    invalid = [c for c in categories if c not in available]
    if invalid:
        print(f"Invalid categories: {invalid}")
        print("Use --list-categories to see valid options.")
        sys.exit(1)

    routes_out_dir = "" if args.no_routes else args.routes_out_dir

    show_pipeline = args.show_pipeline or args.all_prints
    verbose = args.verbose or args.all_prints

    config = SchemaGenerationLoopConfig(
        categories=categories,
        variants_per_category=args.count_per_combination,
        output_dir=args.output_dir,
        town=args.town,
        highway_town=args.highway_town,
        t_junction_town=args.t_junction_town,
        two_lane_corridor_town=args.two_lane_corridor_town,
        model_id=args.model,
        viz_objects=True,
        routes_out_dir=routes_out_dir,
        routes_ego_num=args.routes_ego_num,
        align_routes=not args.no_align_routes,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        max_retries_per_scenario=args.max_retries,
        min_validation_score=args.min_score,
        mute_pipeline=not show_pipeline,
        template_only=args.template_only,
        verbose=verbose,
        allow_template_fallback=not args.no_template_fallback,
    )

    print()
    print("=" * 60)
    print("Schema Scenario Generation")
    print("=" * 60)
    print(f"Model:            {config.model_id}")
    print(f"Categories:       {len(categories)}")
    print(f"Variants/category:{config.variants_per_category}")
    print(f"Output dir:       {config.output_dir}")
    print(f"Town (default):   {config.town}")
    print(f"Town (highway):   {config.highway_town or config.town}")
    print(f"Town (t-junction):{config.t_junction_town or config.town}")
    print(f"Town (2-lane):    {config.two_lane_corridor_town or config.town}")
    print(f"Template only:    {config.template_only}")
    print(f"Routes out dir:   {config.routes_out_dir or 'disabled'}")
    print(f"Align routes:     {config.align_routes} (CARLA {config.carla_host}:{config.carla_port})")
    print("=" * 60)
    print()

    loop = SchemaGenerationLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
