#!/usr/bin/env python

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from simulation.leaderboard.leaderboard.recovery.checkpoint_manager import CheckpointManager
from simulation.leaderboard.leaderboard.recovery.evaluation_state_manager import EvaluationStateManager
from simulation.leaderboard.leaderboard.recovery.planner_continuity import PlannerContinuityManager
from simulation.leaderboard.leaderboard.recovery.provenance_logger import (
    FrameProvenanceLogger,
    RecoveryJournal,
)


class _DummyAgent:
    def __init__(self) -> None:
        self._checkpoint_last_input_data = {"foo": 1}
        self._checkpoint_last_timestamp = 1.0
        self._checkpoint_last_output = {"bar": 2}
        self._world = None
        self._client = None
        self._vehicle = None
        self.rebound = False
        self._state = {"counter": 7}

    def get_checkpoint_state(self):
        return dict(self._state)

    def set_checkpoint_state(self, state):
        self._state = dict(state)

    def rebind_after_recovery(self, recovery_context):
        self.rebound = True
        self._world = recovery_context.get("world")
        self._client = recovery_context.get("client")
        egos = recovery_context.get("ego_vehicles", [])
        self._vehicle = egos[0] if egos else None
        return True


class RecoveryComponentTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_chaining(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                tmpdir,
                checkpoint_interval_frames=5,
                max_checkpoints=0,
            )
            self.assertTrue(mgr.should_checkpoint(0))
            ref1 = mgr.commit(
                {"logical_frame_id": 0, "value": "a"},
                logical_frame_id=0,
                route_name="routeA",
                route_index=0,
                crash_generation=0,
                segment_id=1,
            )
            self.assertEqual(ref1.index, 1)
            self.assertFalse(mgr.should_checkpoint(2))
            self.assertTrue(mgr.should_checkpoint(5))
            ref2 = mgr.commit(
                {"logical_frame_id": 5, "value": "b"},
                logical_frame_id=5,
                route_name="routeA",
                route_index=0,
                crash_generation=1,
                segment_id=2,
            )
            self.assertEqual(ref2.index, 2)
            latest_ref, latest_payload = mgr.load_latest_payload()
            self.assertIsNotNone(latest_ref)
            self.assertEqual(latest_ref.index, 2)
            self.assertEqual(latest_payload["value"], "b")
            mgr.close()

    def test_checkpoint_async_enqueue_and_drain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                tmpdir,
                checkpoint_interval_frames=1,
                max_checkpoints=0,
                async_writes=True,
                queue_size=1,
            )
            scheduled = mgr.enqueue_commit(
                {"logical_frame_id": 0, "value": "async"},
                logical_frame_id=0,
                route_name="routeA",
                route_index=0,
                crash_generation=0,
                segment_id=1,
                reason="periodic",
                snapshot_prepare_ms=0.1,
            )
            self.assertTrue(scheduled["scheduled"])
            found_success = False
            for _ in range(20):
                results = mgr.drain_completed()
                if any(result.ok for result in results):
                    found_success = True
                    break
                import time
                time.sleep(0.05)
            self.assertTrue(found_success)
            latest_ref, latest_payload = mgr.load_latest_payload()
            self.assertIsNotNone(latest_ref)
            self.assertEqual(latest_payload["value"], "async")
            mgr.close()

    def test_recovery_journal_and_frame_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recovery_dir = Path(tmpdir) / "recovery"
            journal = RecoveryJournal(recovery_dir)
            journal.append("crash_detected", route_name="routeA", checkpoint_id=3)

            provenance = FrameProvenanceLogger(recovery_dir)
            provenance.start_segment(checkpoint_id=None, crash_generation=0)
            provenance.log_frame(
                logical_frame_id=10,
                checkpoint_id=None,
                crash_generation=0,
                carla_episode_frame=200,
                frame_origin="original",
            )
            provenance.start_segment(checkpoint_id=3, crash_generation=1, planner_mode="preserved_with_rebind")
            provenance.log_frame(
                logical_frame_id=11,
                checkpoint_id=3,
                crash_generation=1,
                carla_episode_frame=4,
                frame_origin="recovered",
            )

            journal_lines = (recovery_dir / "recovery_journal.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(journal_lines), 1)
            with (recovery_dir / "frame_provenance.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["checkpoint_id"], "3")
            self.assertEqual(rows[1]["frame_origin"], "recovered")

    def test_planner_continuity_modes(self):
        manager = PlannerContinuityManager(replay_window=4, recovery_mode="engineering")
        agent = _DummyAgent()
        manager.observe_runtime(agent)
        snapshot = manager.snapshot(agent)
        mode, notes = manager.restore(
            agent,
            snapshot,
            world="world-handle",
            client="client-handle",
            ego_vehicles=["ego0"],
        )
        self.assertIn(mode, ("preserved_with_rebind", "preserved_alive"))
        self.assertTrue(agent.rebound)
        self.assertEqual(agent._world, "world-handle")
        self.assertIsInstance(notes, list)

    def test_publication_safe_disables_replay_buffer(self):
        manager = PlannerContinuityManager(replay_window=4, recovery_mode="publication_safe")
        agent = _DummyAgent()
        manager.observe_runtime(agent)
        snapshot = manager.snapshot(agent)
        self.assertEqual(snapshot.get("replay_buffer"), [])
        mode, _notes = manager.restore(
            agent,
            snapshot,
            world="world-handle",
            client="client-handle",
            ego_vehicles=["ego0"],
        )
        self.assertIn(mode, ("preserved_with_rebind", "preserved_alive"))

    def test_criterion_state_roundtrip(self):
        class DummyCriterion:
            def __init__(self):
                self.name = "DummyCriterion"
                self.test_status = "RUNNING"
                self.actual_value = 3.5
                self.list_traffic_events = [{"event": "x"}]
                self._private_counter = 9

        class DummyTimeout:
            def __init__(self):
                self.timeout = False
                self._start_time = 1.2

        class DummyScenario:
            def __init__(self):
                self._criteria = [DummyCriterion()]
                self.timeout_node = DummyTimeout()

            def get_criteria(self):
                return self._criteria

        class DummyManager:
            def __init__(self):
                self.scenario = [DummyScenario()]

        manager = DummyManager()
        state_mgr = EvaluationStateManager()
        snapshot = state_mgr._capture_criteria_state(manager)

        criterion = manager.scenario[0].get_criteria()[0]
        criterion.test_status = "FAILURE"
        criterion.actual_value = 99.0
        criterion._private_counter = -1
        manager.scenario[0].timeout_node.timeout = True

        partial = state_mgr._restore_criteria_state(manager, snapshot)
        self.assertFalse(partial)
        self.assertEqual(criterion.test_status, "RUNNING")
        self.assertAlmostEqual(float(criterion.actual_value), 3.5)
        self.assertEqual(int(criterion._private_counter), 9)
        self.assertFalse(manager.scenario[0].timeout_node.timeout)


if __name__ == "__main__":
    unittest.main()
