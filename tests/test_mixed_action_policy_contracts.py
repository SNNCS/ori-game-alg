import unittest

import torch

import config
from agent import CognitiveAgent
from generic_adapter import GenericGameAdapter
from interpretation import build_context
from runtime import RuntimeSnapshot
from specs import ULTIMATUM_SPEC


class TrackingAdapter(GenericGameAdapter):
    def __init__(self, spec):
        super().__init__(spec)
        self.affordance_calls = []

    def action_affordance(self, state=None, actor=None):
        self.affordance_calls.append((state, actor))
        return super().action_affordance(state=state, actor=actor)


class MixedActionPolicyContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(config.SEED)

    def test_generated_policy_output_respects_legal_candidate_mask(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        agent = CognitiveAgent(adapter=adapter)
        snapshot = RuntimeSnapshot(
            state=adapter.initial_tree_state(),
            current_actor=adapter.focal_actor,
            legal_action_mask={
                adapter.focal_actor: (False, True, True, True, True),
            },
        )

        generated = agent.generate_candidate_interventions(snapshot=snapshot)
        policy = generated.policy_output
        probs = policy.action_probs

        self.assertIsNotNone(policy)
        self.assertAlmostEqual(float(probs[0]), 0.0, places=5)
        self.assertNotEqual(int(policy.selected_index), 0)
        self.assertEqual(policy.candidate_logits.shape, (config.N_GENERATED_ACTIONS,))
        self.assertTrue(policy.continuous_params.requires_grad)

        action_event = adapter.ground_action(policy, snapshot=snapshot)
        self.assertIsNotNone(action_event.log_prob)
        self.assertTrue(adapter.validate_intervention(action_event.action))

    def test_all_false_legal_candidate_mask_is_invalid(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        agent = CognitiveAgent(adapter=adapter)
        snapshot = RuntimeSnapshot(
            state=adapter.initial_tree_state(),
            current_actor=adapter.focal_actor,
            legal_action_mask={
                adapter.focal_actor: (False, False, False, False, False),
            },
        )

        with self.assertRaises(ValueError):
            agent.generate_candidate_interventions(snapshot=snapshot)

    def test_action_event_log_prob_matches_decision_policy(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        agent = CognitiveAgent(adapter=adapter)
        snapshot = agent.runtime_snapshot()

        action_event = agent.act(
            snapshot=snapshot,
            context=build_context(prev_reject_rate=0.3, status_gap=-0.2),
        )
        decision = agent._last_decision
        expected = torch.log(
            decision.action_probs[decision.selected_index] + 1e-8)

        self.assertTrue(torch.allclose(action_event.log_prob, expected))
        self.assertEqual(action_event.metadata["policy_source"], "decision")

    def test_decision_selection_respects_legal_candidate_mask(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        agent = CognitiveAgent(adapter=adapter)
        snapshot = RuntimeSnapshot(
            state=adapter.initial_tree_state(),
            current_actor=adapter.focal_actor,
            legal_action_mask={
                adapter.focal_actor: (True, False, False, False, False),
            },
        )

        action_event = agent.act(snapshot=snapshot)

        self.assertEqual(action_event.metadata["decision_index"], 0)

    def test_direct_generation_uses_snapshot_current_actor(self):
        adapter = TrackingAdapter(ULTIMATUM_SPEC)
        agent = CognitiveAgent(adapter=adapter)
        snapshot = RuntimeSnapshot(
            state=adapter.initial_tree_state(),
            current_actor=adapter.counterpart_actor,
        )

        agent.generate_candidate_interventions(snapshot=snapshot)

        state, actor = adapter.affordance_calls[-1]
        self.assertIs(state, snapshot)
        self.assertEqual(actor, adapter.counterpart_actor)

    def test_supplied_candidates_still_receive_runtime_legal_mask(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        agent = CognitiveAgent(adapter=adapter)
        snapshot = RuntimeSnapshot(
            state=adapter.initial_tree_state(),
            current_actor=adapter.focal_actor,
            legal_action_mask={
                adapter.focal_actor: (True, False, False, False, False),
            },
        )
        candidates = [
            adapter.decode_action(torch.tensor([float(idx)]))
            for idx in range(config.N_GENERATED_ACTIONS)
        ]

        decision = agent.deliberate(
            snapshot=snapshot,
            candidate_actions=candidates,
            context=build_context(prev_reject_rate=0.1),
        )

        self.assertIsNotNone(decision.legal_mask)
        self.assertEqual(decision.selected_index, 0)
        self.assertEqual(
            decision.legal_mask.tolist(),
            [True, False, False, False, False],
        )


if __name__ == "__main__":
    unittest.main()
