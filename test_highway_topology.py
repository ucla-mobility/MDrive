#!/usr/bin/env python3
"""Quick test to verify HIGHWAY topology support was added correctly."""

import sys
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scenario_generator"))
sys.path.insert(0, str(REPO_ROOT / "scenario_generator" / "pipeline"))

from scenario_generator.capabilities import (
    CATEGORY_DEFINITIONS,
    TopologyType,
    PIPELINE_CAPABILITIES,
    get_available_categories,
)

def test_highway_topology():
    print("=" * 80)
    print("Testing HIGHWAY Topology Support")
    print("=" * 80)
    
    # Test 1: Check TopologyType enum has HIGHWAY
    print("\n1. Checking TopologyType enum...")
    assert hasattr(TopologyType, 'HIGHWAY'), "HIGHWAY not found in TopologyType enum"
    assert TopologyType.HIGHWAY.value == "highway", f"HIGHWAY value is {TopologyType.HIGHWAY.value}, expected 'highway'"
    print("   ✓ TopologyType.HIGHWAY exists with value 'highway'")
    
    # Test 2: Check HIGHWAY in supported topologies
    print("\n2. Checking supported topologies...")
    assert TopologyType.HIGHWAY in PIPELINE_CAPABILITIES.supported_topologies, \
        "HIGHWAY not in supported_topologies"
    print(f"   ✓ HIGHWAY in supported topologies: {[t.value for t in PIPELINE_CAPABILITIES.supported_topologies]}")
    
    # Test 3: Check highway categories exist and use correct topology
    print("\n3. Checking highway categories...")
    highway_categories = [
        "Highway On-Ramp Merge",
        "Highway Weaving",
    ]
    
    for cat_name in highway_categories:
        assert cat_name in CATEGORY_DEFINITIONS, f"Category '{cat_name}' not found"
        cat_def = CATEGORY_DEFINITIONS[cat_name]
        assert cat_def.map.topology == TopologyType.HIGHWAY, \
            f"Category '{cat_name}' has topology {cat_def.map.topology}, expected HIGHWAY"
        print(f"   ✓ '{cat_name}' uses TopologyType.HIGHWAY")
        print(f"     - needs_on_ramp: {cat_def.map.needs_on_ramp}")
        print(f"     - needs_merge: {cat_def.map.needs_merge}")
        print(f"     - needs_multi_lane: {cat_def.map.needs_multi_lane}")
    
    # Test 4: Check CropFeatures has is_highway field
    print("\n4. Checking CropFeatures model...")
    sys.path.insert(0, str(REPO_ROOT / "scenario_generator" / "pipeline"))
    from step_01_crop.models import CropFeatures
    import inspect
    
    sig = inspect.signature(CropFeatures.__init__)
    params = list(sig.parameters.keys())
    assert 'is_highway' in params, "is_highway not found in CropFeatures parameters"
    print("   ✓ CropFeatures has 'is_highway' field")
    
    # Test 5: Test GeometrySpec generation with HIGHWAY
    print("\n5. Testing GeometrySpec generation with HIGHWAY topology...")
    from scenario_generator.constraints import ScenarioSpec, EgoVehicleSpec, EgoManeuver
    from scenario_generator.schema_utils import geometry_spec_from_scenario_spec
    
    test_spec = ScenarioSpec(
        category="Highway On-Ramp Merge",
        topology=TopologyType.HIGHWAY,
        needs_oncoming=False,
        needs_multi_lane=True,
        needs_on_ramp=True,
        needs_merge=True,
        ego_vehicles=[
            EgoVehicleSpec(
                vehicle_id="Vehicle 1",
                maneuver=EgoManeuver.STRAIGHT,
                lane_change_phase="unknown",
                entry_road="unknown",
                exit_road="unknown",
            )
        ],
        vehicle_constraints=[],
        actors=[],
    )
    
    geo_spec = geometry_spec_from_scenario_spec(test_spec)
    assert geo_spec.topology == "highway", f"GeometrySpec topology is {geo_spec.topology}, expected 'highway'"
    assert geo_spec.min_lane_count >= 3, f"Highway min_lane_count is {geo_spec.min_lane_count}, expected >= 3"
    print(f"   ✓ GeometrySpec correctly generated:")
    print(f"     - topology: {geo_spec.topology}")
    print(f"     - min_lane_count: {geo_spec.min_lane_count}")
    print(f"     - needs_on_ramp: {geo_spec.needs_on_ramp}")
    print(f"     - needs_merge_onto_same_road: {geo_spec.needs_merge_onto_same_road}")
    
    # Test 6: Verify available categories includes highway ones
    print("\n6. Checking available categories...")
    all_categories = get_available_categories()
    for cat_name in highway_categories:
        assert cat_name in all_categories, f"'{cat_name}' not in available categories"
    print(f"   ✓ All highway categories available (total: {len(all_categories)} categories)")
    
    # Test 7: Visual verification - Generate highway crops from Town06
    print("\n7. Generating highway visualizations from Town06...")
    generate_highway_visualizations()
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - HIGHWAY topology support successfully added!")
    print("=" * 80)


def generate_highway_visualizations():
    """Generate visualizations of detected highway crops from Town06."""
    import os
    import json
    from step_01_crop.candidates import build_candidate_crops_for_town
    
    # Town06 should have highway geometry
    town_json_path = REPO_ROOT / "scenario_generator" / "town_nodes" / "Town06.json"
    
    if not town_json_path.exists():
        print(f"   ⚠ Town06.json not found at {town_json_path}, skipping visualization")
        return
    
    print(f"   Building crop candidates from Town06...")
    crops = build_candidate_crops_for_town(
        town_name="Town06",
        town_json_path=str(town_json_path),
        radii=[55.0, 65.0],  # Medium to large crops for highways
        min_path_len=30.0,   # Longer paths for highways
        max_paths=100,
        max_depth=10,
    )
    
    print(f"   Total crops generated: {len(crops)}")
    
    # Filter to highway crops
    highway_crops = [c for c in crops if c.is_highway]
    corridor_crops = [c for c in crops if not c.is_highway and not c.is_t_junction and not c.is_four_way]
    
    print(f"   Highway crops detected: {len(highway_crops)}")
    print(f"   Non-highway corridors: {len(corridor_crops)}")
    
    if not highway_crops:
        print("   ⚠ No highway crops detected in Town06")
        return
    
    # Create output directory
    viz_dir = REPO_ROOT / "highway_visualization_test"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate visualizations for top highway crops
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("   ⚠ matplotlib not available, skipping visualization generation")
        return
    
    from step_01_crop.viz import save_viz
    
    # Sort by lane count (descending) to get the most highway-like ones first
    highway_crops.sort(key=lambda c: c.lane_count_est, reverse=True)
    
    max_viz = 5
    print(f"   Generating visualizations for top {min(max_viz, len(highway_crops))} highway crops...")
    
    for i, crop in enumerate(highway_crops[:max_viz], start=1):
        # Calculate straight ratio
        if crop.maneuver_stats.get('straight', {}).get('count', 0) > 0:
            total_paths = sum(crop.maneuver_stats.get(m, {}).get('count', 0) 
                            for m in ['straight', 'left', 'right', 'uturn'])
            straight_ratio = crop.maneuver_stats['straight']['count'] / max(1, total_paths)
        else:
            straight_ratio = 0.0
        
        scenario_text = (
            f"Highway Detection Test #{i} | "
            f"Lanes: {crop.lane_count_est} | "
            f"Straight: {straight_ratio:.1%} | "
            f"Junctions: {crop.junction_count} | "
            f"Paths: {crop.n_paths} | "
            f"Has merge: {crop.has_merge_onto_same_road} | "
            f"Has on-ramp: {crop.has_on_ramp}"
        )
        
        out_path = viz_dir / f"highway_crop_{i:02d}_lanes{crop.lane_count_est}.png"
        
        try:
            save_viz(
                out_png=str(out_path),
                scenario_id=f"Highway Test {i}",
                scenario_text=scenario_text,
                crop=crop.crop,
                crop_feat=crop,
                invert_x=False,
                dpi=120,
            )
            print(f"      ✓ Saved: {out_path.name}")
        except Exception as e:
            print(f"      ✗ Failed to save {out_path.name}: {e}")
    
    # Also save a few non-highway corridors for comparison
    if corridor_crops:
        corridor_crops.sort(key=lambda c: c.lane_count_est, reverse=True)
        print(f"   Generating comparison visualizations for {min(3, len(corridor_crops))} non-highway corridors...")
        
        for i, crop in enumerate(corridor_crops[:3], start=1):
            if crop.maneuver_stats.get('straight', {}).get('count', 0) > 0:
                total_paths = sum(crop.maneuver_stats.get(m, {}).get('count', 0) 
                                for m in ['straight', 'left', 'right', 'uturn'])
                straight_ratio = crop.maneuver_stats['straight']['count'] / max(1, total_paths)
            else:
                straight_ratio = 0.0
            
            scenario_text = (
                f"Non-Highway Corridor #{i} | "
                f"Lanes: {crop.lane_count_est} | "
                f"Straight: {straight_ratio:.1%} | "
                f"Junctions: {crop.junction_count} | "
                f"Paths: {crop.n_paths}"
            )
            
            out_path = viz_dir / f"corridor_crop_{i:02d}_lanes{crop.lane_count_est}.png"
            
            try:
                save_viz(
                    out_png=str(out_path),
                    scenario_id=f"Corridor Test {i}",
                    scenario_text=scenario_text,
                    crop=crop.crop,
                    crop_feat=crop,
                    invert_x=False,
                    dpi=120,
                )
                print(f"      ✓ Saved: {out_path.name}")
            except Exception as e:
                print(f"      ✗ Failed to save {out_path.name}: {e}")
    
    # Save summary stats
    summary_path = viz_dir / "detection_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Highway Detection Summary for Town06\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total crops: {len(crops)}\n")
        f.write(f"Highway crops: {len(highway_crops)} ({100*len(highway_crops)/max(1,len(crops)):.1f}%)\n")
        f.write(f"Non-highway corridors: {len(corridor_crops)}\n")
        f.write(f"Intersections: {sum(1 for c in crops if c.is_t_junction or c.is_four_way)}\n\n")
        
        f.write("Top 5 Highway Crops:\n")
        f.write("-" * 80 + "\n")
        for i, crop in enumerate(highway_crops[:5], start=1):
            total_paths = sum(crop.maneuver_stats.get(m, {}).get('count', 0) 
                            for m in ['straight', 'left', 'right', 'uturn'])
            straight_ratio = crop.maneuver_stats['straight']['count'] / max(1, total_paths)
            f.write(f"\n{i}. Crop at {crop.crop.to_str()}\n")
            f.write(f"   - Lanes: {crop.lane_count_est}\n")
            f.write(f"   - Straight ratio: {straight_ratio:.1%}\n")
            f.write(f"   - Junction count: {crop.junction_count}\n")
            f.write(f"   - Total paths: {crop.n_paths}\n")
            f.write(f"   - Has merge: {crop.has_merge_onto_same_road}\n")
            f.write(f"   - Has on-ramp: {crop.has_on_ramp}\n")
    
    print(f"\n   ✓ Visualizations saved to: {viz_dir}/")
    print(f"   ✓ Summary saved to: {summary_path.name}")


if __name__ == "__main__":
    try:
        test_highway_topology()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
