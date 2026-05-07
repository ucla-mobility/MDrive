"""Unit tests for recursive scenario discovery and XML override parsing.

Covers:
- ``detect_scenario_subfolders``: walks arbitrarily nested folders (e.g.
  ``interaction/<Category>/<N>/``) and returns a scenario leaf whenever a directory
  has ego XMLs directly inside it.
- ``read_scenario_actor_overrides``: extracts ``actor_control_mode`` and
  ``log_replay_actors`` attributes declared on the ``<routes>`` element (or
  ``<route>`` element) of any ego XML in the scenario dir.

Run with: ``python -m unittest tools.tests.test_scenario_discovery``
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _write_ego_xml(path: Path, *, routes_attrs: str = "", route_attrs: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <routes{(" " + routes_attrs) if routes_attrs else ""}>
          <route id="r0" role="ego" town="Town01"{(" " + route_attrs) if route_attrs else ""}>
            <waypoint x="0" y="0" z="0" pitch="0" yaw="0" roll="0"/>
            <waypoint x="1" y="0" z="0" pitch="0" yaw="0" roll="0"/>
          </route>
        </routes>
        """
    )
    path.write_text(xml, encoding="utf-8")


class DetectScenarioSubfoldersTests(unittest.TestCase):
    def test_routes_dir_itself_is_scenario_returns_empty(self):
        from tools.run_custom_eval import detect_scenario_subfolders

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(root / "ego.xml")
            self.assertEqual(detect_scenario_subfolders(root), [])

    def test_single_level_layout_v2xpnp_like(self):
        from tools.run_custom_eval import detect_scenario_subfolders

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(root / "s1" / "ego.xml")
            _write_ego_xml(root / "s2" / "ego.xml")
            result = detect_scenario_subfolders(root)
            names = sorted(n for _, n in result)
            self.assertEqual(names, ["s1", "s2"])

    def test_two_level_layout_interaction_like(self):
        from tools.run_custom_eval import detect_scenario_subfolders

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(root / "Blocked_Lane" / "1" / "ego.xml")
            _write_ego_xml(root / "Blocked_Lane" / "2" / "ego.xml")
            _write_ego_xml(root / "Roundabout" / "1" / "ego.xml")
            result = detect_scenario_subfolders(root)
            names = sorted(n for _, n in result)
            self.assertEqual(
                names,
                ["Blocked_Lane/1", "Blocked_Lane/2", "Roundabout/1"],
            )

    def test_mixed_depth_layout(self):
        from tools.run_custom_eval import detect_scenario_subfolders

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(root / "v2xpnp" / "scenario_a" / "ego.xml")
            _write_ego_xml(root / "interaction" / "Cat" / "1" / "ego.xml")
            _write_ego_xml(root / "opencda" / "A" / "ego.xml")
            result = detect_scenario_subfolders(root)
            names = sorted(n for _, n in result)
            self.assertEqual(
                names,
                ["interaction/Cat/1", "opencda/A", "v2xpnp/scenario_a"],
            )

    def test_actors_subfolder_is_skipped(self):
        from tools.run_custom_eval import detect_scenario_subfolders

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(root / "scenario_x" / "ego.xml")
            # actors/static containing XMLs must not be treated as a scenario.
            (root / "scenario_x" / "actors" / "static").mkdir(parents=True)
            _write_ego_xml(root / "scenario_x" / "actors" / "static" / "child.xml")
            result = detect_scenario_subfolders(root)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][1], "scenario_x")

    def test_no_descent_past_scenario_leaf(self):
        """Once a scenario leaf is found, deeper dirs must not produce more scenarios."""
        from tools.run_custom_eval import detect_scenario_subfolders

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(root / "s1" / "ego.xml")
            _write_ego_xml(root / "s1" / "nested_sub" / "ego.xml")  # should NOT appear
            result = detect_scenario_subfolders(root)
            names = [n for _, n in result]
            self.assertEqual(names, ["s1"])

    def test_non_ego_role_xml_ignored(self):
        from tools.run_custom_eval import detect_scenario_subfolders

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Only a non-ego XML at the top -> not a scenario, no subdirs -> no result.
            path = root / "npc.xml"
            path.write_text(
                '<?xml version="1.0" encoding="utf-8"?>\n'
                '<routes><route id="r" role="npc" town="Town01">'
                '<waypoint x="0" y="0" z="0" pitch="0" yaw="0" roll="0"/>'
                '<waypoint x="1" y="0" z="0" pitch="0" yaw="0" roll="0"/>'
                '</route></routes>\n',
                encoding="utf-8",
            )
            result = detect_scenario_subfolders(root)
            self.assertEqual(result, [])


class ReadScenarioActorOverridesTests(unittest.TestCase):
    def test_no_overrides_returns_empty(self):
        from tools.run_custom_eval import read_scenario_actor_overrides

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(root / "ego.xml")
            self.assertEqual(read_scenario_actor_overrides(root), {})

    def test_attrs_on_routes_element(self):
        from tools.run_custom_eval import read_scenario_actor_overrides

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(
                root / "ego.xml",
                routes_attrs='actor_control_mode="replay" log_replay_actors="true"',
            )
            result = read_scenario_actor_overrides(root)
            self.assertEqual(result.get("actor_control_mode"), "replay")
            self.assertEqual(result.get("log_replay_actors"), True)

    def test_attrs_on_route_element(self):
        from tools.run_custom_eval import read_scenario_actor_overrides

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(
                root / "ego.xml",
                route_attrs='actor_control_mode="policy" log_replay_actors="false"',
            )
            result = read_scenario_actor_overrides(root)
            self.assertEqual(result.get("actor_control_mode"), "policy")
            self.assertEqual(result.get("log_replay_actors"), False)

    def test_invalid_mode_ignored(self):
        from tools.run_custom_eval import read_scenario_actor_overrides

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(
                root / "ego.xml",
                routes_attrs='actor_control_mode="bogus"',
            )
            self.assertNotIn("actor_control_mode", read_scenario_actor_overrides(root))

    def test_multiple_xmls_first_wins(self):
        from tools.run_custom_eval import read_scenario_actor_overrides

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # sorted() ordering means a_*.xml is visited before z_*.xml.
            _write_ego_xml(
                root / "a_ego.xml",
                routes_attrs='actor_control_mode="replay"',
            )
            _write_ego_xml(
                root / "z_ego.xml",
                routes_attrs='actor_control_mode="policy"',
            )
            result = read_scenario_actor_overrides(root)
            self.assertEqual(result.get("actor_control_mode"), "replay")


class ApplyScenarioXmlOverridesTests(unittest.TestCase):
    def _fake_args(self, *, mode="policy", log_replay=False):
        import argparse

        return argparse.Namespace(
            custom_actor_control_mode=mode,
            log_replay_actors=log_replay,
        )

    def test_applies_override_from_xml(self):
        from tools.run_custom_eval import apply_scenario_xml_overrides

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_ego_xml(
                root / "ego.xml",
                routes_attrs='actor_control_mode="replay" log_replay_actors="true"',
            )
            args = self._fake_args()
            apply_scenario_xml_overrides(args, root)
            self.assertEqual(args.custom_actor_control_mode, "replay")
            self.assertEqual(args.log_replay_actors, True)

    def test_restore_defaults_between_iterations(self):
        from tools.run_custom_eval import apply_scenario_xml_overrides

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sc_replay = root / "replay_sc"
            sc_policy = root / "policy_sc"
            _write_ego_xml(
                sc_replay / "ego.xml",
                routes_attrs='actor_control_mode="replay" log_replay_actors="true"',
            )
            _write_ego_xml(sc_policy / "ego.xml")  # no override

            args = self._fake_args(mode="policy", log_replay=False)
            saved = apply_scenario_xml_overrides(args, sc_replay)
            self.assertEqual(args.custom_actor_control_mode, "replay")
            self.assertEqual(args.log_replay_actors, True)
            apply_scenario_xml_overrides(args, sc_policy, saved_defaults=saved)
            # sc_policy has no override, should restore CLI defaults.
            self.assertEqual(args.custom_actor_control_mode, "policy")
            self.assertEqual(args.log_replay_actors, False)


if __name__ == "__main__":
    unittest.main()
