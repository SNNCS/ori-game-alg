import unittest

import torch

import config
from agent import CognitiveAgent
from generic_adapter import GenericGameAdapter
from interpretation import build_context
from specs.ultimatum import build_ultimatum_spec


class AdapterAndDecisionContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(config.SEED)

    def test_agent_generates_latent_actions_before_adapter_grounding(self):
        agent = CognitiveAgent()
        context = build_context(turn_idx=1, session_len=8)
        snapshot = agent.runtime_snapshot()

        generated = agent.generate_candidate_interventions(
            context=context, snapshot=snapshot)
        affordance = agent.adapter.action_affordance()

        self.assertEqual(
            len(generated.candidates),
            config.N_GENERATED_ACTIONS,
        )
        self.assertEqual(
            generated.latent_actions.shape,
            (config.N_GENERATED_ACTIONS, config.ACTION_LATENT_DIM),
        )
        self.assertTrue(generated.latent_actions.requires_grad)

        for candidate in generated.candidates:
            self.assertEqual(candidate.metadata["source"], "generated")
            self.assertIsNotNone(candidate.latent_action)
            self.assertTrue(candidate.latent_action.requires_grad)
            action_value = float(candidate.action)
            self.assertGreaterEqual(action_value, affordance.low)
            self.assertLessEqual(action_value, affordance.high)
            self.assertTrue(agent.adapter.validate_intervention(candidate.action))

    def test_adapter_response_count_drives_history_and_branch_policy_sizes(self):
        adapter = GenericGameAdapter(build_ultimatum_spec(
            responses=("accept", "reject", "counter", "delay")))
        agent = CognitiveAgent(adapter=adapter)
        context = build_context(turn_idx=0, session_len=8)
        snapshot = agent.runtime_snapshot()

        agent.act(snapshot=snapshot, context=context)
        selected_tree = agent._last_decision.selected_future.tree

        self.assertEqual(agent.tree.policy.n_responses, 4)
        self.assertEqual(agent.history.W_enc.in_features, config.M + 4 + config.D)
        self.assertEqual(len(selected_tree.children), 4)

    def test_decision_scores_all_candidate_futures(self):
        agent = CognitiveAgent()
        context = build_context(turn_idx=2, session_len=8)
        snapshot = agent.runtime_snapshot()

        decision = agent.deliberate(snapshot=snapshot, context=context)

        self.assertEqual(len(decision.futures), config.N_GENERATED_ACTIONS)
        self.assertEqual(decision.scores.shape, (config.N_GENERATED_ACTIONS,))
        self.assertEqual(decision.action_probs.shape, (config.N_GENERATED_ACTIONS,))
        self.assertAlmostEqual(float(decision.action_probs.sum()), 1.0, places=5)
        self.assertTrue(torch.isfinite(decision.expected_utility))
        self.assertIs(decision.selected_future, decision.futures[decision.selected_index])


if __name__ == "__main__":
    unittest.main()
