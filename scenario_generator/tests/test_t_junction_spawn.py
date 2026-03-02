import json
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scenario_generator.pipeline.step_04_path_refiner.geometry import (
    _build_segment_payload_from_polyline,
)
from scenario_generator.pipeline.step_04_path_refiner.main import refine_picked_paths_with_model


def _segment(points, seg_id):
    return _build_segment_payload_from_polyline(points, seg_id=seg_id)


class TJunctionSpawnTest(unittest.TestCase):
    def test_perpendicular_side_road_spawn_not_clipped(self):
        """
        Regression: for t-junctions with perpendicular_left_of, the side-road vehicle
        should keep its off-crop spawn instead of being clipped onto the main corridor.
        """
        crop = {"xmin": -5.0, "xmax": 5.0, "ymin": -10.0, "ymax": 30.0}

        # Vehicle 1: approaches from side road (west -> east, then left turn north)
        v1_segments = [
            _segment([(-15.0, 0.0), (-5.0, 0.0)], seg_id=101),
            _segment([(-5.0, 0.0), (0.0, 0.0), (0.0, 20.0)], seg_id=102),
        ]
        # Vehicle 2: main road straight southbound
        v2_segments = [
            _segment([(0.0, 25.0), (0.0, -20.0)], seg_id=201),
        ]

        payload = {
            "nodes": "",
            "crop_region": crop,
            "picked": [
                {
                    "vehicle": "Vehicle 1",
                    "name": "side_left",
                    "signature": {
                        "entry": {"point": {"x": -15.0, "y": 0.0}},
                        "exit": {"point": {"x": 0.0, "y": 20.0}},
                        "segments_detailed": v1_segments,
                    },
                },
                {
                    "vehicle": "Vehicle 2",
                    "name": "main_straight",
                    "signature": {
                        "entry": {"point": {"x": 0.0, "y": 25.0}},
                        "exit": {"point": {"x": 0.0, "y": -20.0}},
                        "segments_detailed": v2_segments,
                    },
                },
            ],
        }

        # Schema with perpendicular + merge constraints; side-road vehicle should not get a merge macro
        schema_payload = {
            "topology": "t_junction",
            "ego_vehicles": [
                {"vehicle_id": "Vehicle 1", "entry_road": "side", "exit_road": "main", "maneuver": "left"},
                {"vehicle_id": "Vehicle 2", "entry_road": "main", "exit_road": "main", "maneuver": "straight"},
            ],
            "vehicle_constraints": [
                {"a": "Vehicle 1", "b": "Vehicle 2", "type": "perpendicular_left_of"},
                {"a": "Vehicle 1", "b": "Vehicle 2", "type": "merges_into_lane_of"},
                {"a": "Vehicle 2", "b": "Vehicle 1", "type": "merges_into_lane_of"},
            ],
        }

        # Persist under repo to avoid /tmp usage
        out_dir = Path("scenario_generator/tests/artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        picked_path = out_dir / "t_junction_picked_paths.json"
        out_path = out_dir / "t_junction_refined.json"
        picked_path.write_text(json.dumps(payload, indent=2))

        viz_out = os.environ.get("TJUNCTION_TEST_VIZ")
        if viz_out:
            viz_out = str(Path(viz_out))
        else:
            viz_out = str(out_dir / "t_junction_refined.png")
        viz = True  # always write a PNG for this regression test

        refine_picked_paths_with_model(
            picked_paths_json=str(picked_path),
            description="Side-road vehicle turns left into main road; main vehicle goes straight.",
            out_json=str(out_path),
            model=None,
            tokenizer=None,
            schema_payload=schema_payload,
            viz=viz,
            viz_out=viz_out,
        )

        refined = json.loads(out_path.read_text())
        constraints = refined.get("refinement", {}).get("constraints", {})

        # Lane-change macros targeting side-road vehicles should be dropped.
        lane_changes = constraints.get("lane_changes", [])
        self.assertTrue(
            all(lc.get("vehicle") != "Vehicle 1" and lc.get("target") != "Vehicle 1" for lc in lane_changes),
            "Side-road vehicle should not receive lane-change macros in t-junction perpendicular cases.",
        )

        # Spawn should remain on the side road (entry x < -10, i.e., outside the crop box).
        v1 = next(p for p in refined["picked"] if p.get("vehicle") == "Vehicle 1")
        entry_point = v1["signature"]["entry"]["point"]
        self.assertLess(
            entry_point["x"],
            -10.0,
            f"Expected side-road spawn to stay outside crop; got entry {entry_point}",
        )


if __name__ == "__main__":
    unittest.main()
