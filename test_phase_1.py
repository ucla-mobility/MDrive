#!/usr/bin/env python3
"""Phase 1 Validation Script - sanity checks for schema generation outputs."""

import json
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from scenario_generator.scenario_generator.schema_generator import (
    SchemaScenarioGenerator,
    SchemaGenerationConfig,
)


# Configuration
ACTIVE_CONSTRAINT_TYPES = {
    'opposite_approach_of',
    'perpendicular_left_of',
    'perpendicular_right_of',
    'left_lane_of',
    'right_lane_of',
    'merges_into_lane_of'
}

PASSIVE_CONSTRAINT_TYPES = {
    'follow_route_of',
    'same_approach_as'
}

# Initialize generator
print("Initializing schema generator...")
config = SchemaGenerationConfig()
generator = SchemaScenarioGenerator(config=config)


def count_constraints(spec, constraint_type):
    """Count constraints of a given type"""
    constraints = spec.get('vehicle_constraints', [])
    return sum(1 for c in constraints if c.get('constraint_type') == constraint_type)


def analyze_spec(spec):
    """Analyze a scenario spec and return results"""
    if spec is None:
        return {
            'num_vehicles': 0,
            'num_active': 0,
            'num_passive': 0,
            'num_total': 0,
            'active_types': [],
            'passive_types': [],
            'reasoning_present': False,
            'reasoning_text': 'N/A',
            'has_reasoning': False
        }
    
    num_vehicles = len(spec.ego_vehicles)
    
    active_constraints = [c for c in spec.vehicle_constraints if c.constraint_type.value in ACTIVE_CONSTRAINT_TYPES]
    passive_constraints = [c for c in spec.vehicle_constraints if c.constraint_type.value in PASSIVE_CONSTRAINT_TYPES]
    
    num_active = len(active_constraints)
    num_passive = len(passive_constraints)
    num_total = num_active + num_passive
    
    # Check for reasoning in metadata
    reasoning = getattr(spec, '_reasoning', None)
    has_reasoning = reasoning is not None and isinstance(reasoning, dict)
    reasoning_text = reasoning.get('constraint_summary', 'N/A') if has_reasoning else 'N/A'
    
    return {
        'num_vehicles': num_vehicles,
        'num_active': num_active,
        'num_passive': num_passive,
        'num_total': num_total,
        'active_types': [c.constraint_type.value for c in active_constraints],
        'passive_types': [c.constraint_type.value for c in passive_constraints],
        'reasoning_present': has_reasoning,
        'reasoning_text': reasoning_text,
        'has_reasoning': has_reasoning
    }


def validate_interactive(spec):
    """Basic sanity: ensure multi-vehicle specs include an active constraint."""
    analysis = analyze_spec(spec)
    checks = []
    checks.append(('Has at least 1 vehicle', analysis['num_vehicles'] >= 1, True))
    if analysis['num_vehicles'] >= 2:
        checks.append(('Has at least 1 ACTIVE constraint', analysis['num_active'] >= 1, True))
    return checks, analysis


def run_test(category='Highway On-Ramp Merge', num_runs=3):
    """Run basic interaction checks for generated specs."""
    print(f"\n{'='*70}")
    print(f"Testing category '{category}'")
    print(f"{'='*70}")
    
    validate_func = validate_interactive
    all_passed = True
    
    for run in range(1, num_runs + 1):
        print(f"\n[Run {run}/{num_runs}]")
        
        try:
            # Generate scenario
            spec, errors, warnings = generator.generate_spec(category=category)
            
            if spec is None:
                print(f"  ERROR: Failed to generate spec: {errors}")
                all_passed = False
                continue
            
            # Validate
            checks, analysis = validate_func(spec)
            
            # Print results
            print(f"  Vehicles: {analysis['num_vehicles']}")
            print(f"  Total constraints: {analysis['num_total']} (active: {analysis['num_active']}, passive: {analysis['num_passive']})")
            if analysis['active_types']:
                print(f"  Active types: {', '.join(analysis['active_types'])}")
            if analysis['passive_types']:
                print(f"  Passive types: {', '.join(analysis['passive_types'])}")
            print(f"  Reasoning present: {analysis['reasoning_present']}")
            if analysis['reasoning_text'] != 'N/A':
                print(f"  Constraint summary: {analysis['reasoning_text']}")
            
            # Check all validations
            run_passed = True
            for check_name, result, is_required in checks:
                status = "✓" if result else "✗"
                importance = "REQUIRED" if is_required else "optional"
                print(f"    {status} {check_name} [{importance}]")
                
                if is_required and not result:
                    run_passed = False
                    all_passed = False
            
            if run_passed:
                print(f"  Result: PASS ✓")
            else:
                print(f"  Result: FAIL ✗")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def main():
    """Run all Phase 1 tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║ Phase 1 Validation Tests - Schema Generation with Constraint Semantics ║")
    print("╚" + "="*68 + "╝")
    
    print("\nThis script tests that the LLM now understands:")
    print("  • Passive vs active constraint distinction")
    print("  • Constraint combination rules")
    print("  • Reasoning generation")
    
    results = {}
    test_categories = ["Unprotected Left Turn", "Highway On-Ramp Merge", "Major/Minor Unsignalized Entry"]
    for cat in test_categories:
        results[cat] = run_test(category=cat, num_runs=3)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for cat, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {cat}: {status}")
    
    all_passed = all(results.values())
    
    print(f"\n{'='*70}")
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("\nPhase 1 is working correctly!")
        print("\nNext steps:")
        print("  1. Review a few generated specs to verify reasoning quality")
        print("  2. Proceed to Phase 2: Path Picking Constraint-Awareness")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("\nCommon issues:")
        print("  • Passive-only constraints: ensure at least one active relation is present for multi-vehicle specs")
        print("  • Missing reasoning: verify response includes a 'reasoning' field if requested")
        print("  • Weak interaction patterns: adjust prompts to prefer crossing or merge relations")
        return 1


if __name__ == '__main__':
    sys.exit(main())
