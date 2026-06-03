import unittest

import torch

from agent import CognitiveAgent
from experience import response_prediction_loss
from generic_adapter import GenericGameAdapter
from runtime import (
    ActionEvent, CheckpointMetadata, RuntimeSnapshot, RuntimeSchema,
    WorldResponse, check_schema_compatibility,
)
from specs import ULTIMATUM_SPEC


class RuntimeContractTests(unittest.TestCase):
    def test_runtime_transition_uses_current_snapshot_state(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        snapshot = adapter.initial_runtime_snapshot()
        state = dict(snapshot.state)
        state["pie"] = 2.0
        snapshot = RuntimeSnapshot(
            state=state,
            current_actor=adapter.focal_actor,
            public={"pie": 2.0, "paths_open": 1.0},
        )

        action_event = adapter.ground_action(0.7, snapshot=snapshot)
        result = adapter.transition_event(
            snapshot, action_event, WorldResponse("accept"))

        self.assertTrue(result.after.terminal)
        self.assertAlmostEqual(
            float(result.terminal_outcome.payoff_for(adapter.focal_actor)),
            1.4,
            places=5,
        )
        self.assertAlmostEqual(float(snapshot.state["pie"]), 2.0, places=5)

    def test_observation_private_state_is_viewer_relative(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        snapshot = RuntimeSnapshot(
            state={"pie": 1.0, "paths_open": 1.0, "terminal": False},
            current_actor=0,
            public={"pie": 1.0},
            private={
                0: {"hole_strength": 0.9},
                1: {"hole_strength": 0.1},
            },
        )

        obs0 = adapter.encode_observation(snapshot, viewer=0)
        obs1 = adapter.encode_observation(snapshot, viewer=1)

        self.assertEqual(obs0.private_state, {"hole_strength": 0.9})
        self.assertEqual(obs1.private_state, {"hole_strength": 0.1})
        self.assertEqual(obs0.spec.private_feature_names, ("hole_strength",))
        self.assertEqual(obs1.spec.private_feature_names, ("hole_strength",))
        self.assertNotEqual(obs0.private_state, obs1.private_state)
        self.assertEqual(obs0.vector.shape, obs0.mask.shape)
        private_start = len(obs0.spec.feature_names)
        self.assertTrue(torch.allclose(
            obs0.vector[:private_start], obs1.vector[:private_start]))
        self.assertAlmostEqual(float(obs0.vector[private_start]), 0.9, places=5)
        self.assertAlmostEqual(float(obs1.vector[private_start]), 0.1, places=5)

    def test_observation_schema_names_flattened_vector_positions(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        snapshot = RuntimeSnapshot(
            state={"pie": 1.0, "paths_open": 1.0},
            current_actor=0,
            public={"public_vec": torch.tensor([0.2, 0.3])},
            private={0: {"secret_vec": torch.tensor([0.7, 0.8])}},
        )

        obs = adapter.encode_observation(snapshot, viewer=0)

        self.assertIn("public_vec[0]", obs.spec.feature_names)
        self.assertIn("public_vec[1]", obs.spec.feature_names)
        self.assertIn("secret_vec[0]", obs.spec.private_feature_names)
        self.assertIn("secret_vec[1]", obs.spec.private_feature_names)

    def test_unknown_actor_counterpart_is_not_guessed(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)

        with self.assertRaises(ValueError):
            adapter.counterpart_for_actor(adapter.observer_actor)

        snapshot = RuntimeSnapshot(
            state=adapter.initial_tree_state(),
            current_actor=adapter.observer_actor,
        )
        with self.assertRaises(ValueError):
            adapter.response_actor(snapshot=snapshot)

    def test_response_prediction_rejects_action_event_target(self):
        agent = CognitiveAgent()
        snapshot = agent.runtime_snapshot()
        action_event = agent.act(snapshot=snapshot)
        decision = agent._last_decision

        with self.assertRaises(TypeError):
            response_prediction_loss(decision.selected_future, action_event)

        with self.assertRaises(TypeError):
            response_prediction_loss(decision.selected_future, "accept")

    def test_schema_compatibility_reports_mismatches(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        active = adapter.runtime_schema()
        stale = RuntimeSchema(
            spec_name=active.spec_name,
            spec_version=active.spec_version,
            observation_features=("changed",),
            action_controls=active.action_controls,
            world_response_labels=active.world_response_labels,
            outcome_features=active.outcome_features,
            n_entities=active.n_entities,
        )

        report = check_schema_compatibility(
            active, CheckpointMetadata(runtime_schema=stale))

        self.assertFalse(report.compatible)
        self.assertIn("observation_features", report.mismatches)


if __name__ == "__main__":
    unittest.main()
