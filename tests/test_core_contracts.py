import unittest

import torch

import config
from generic_adapter import GenericGameAdapter
from interpretation import build_context, build_signal
from relation_graph import RelationGraph
from runtime import WorldResponse
from specs import ULTIMATUM_SPEC


class CoreContractTests(unittest.TestCase):
    def test_config_and_signal_dimensions_stay_in_sync(self):
        config.sanity_check()

        context = build_context(
            turn_idx=4,
            session_len=20,
            prev_reject_rate=0.25,
            status_gap=0.1,
            urgency=0.3,
        )
        self.assertEqual(context.shape, (config.N_CONTEXT,))

        bid = torch.tensor(0.7, requires_grad=True)
        signal = build_signal(bid, context)

        self.assertEqual(signal.shape, (config.M,))
        self.assertTrue(signal.requires_grad)

        signal.sum().backward()
        self.assertIsNotNone(bid.grad)
        self.assertNotEqual(float(bid.grad), 0.0)

    def test_relation_graph_is_trainable_and_clamped_on_read(self):
        graph = RelationGraph()

        self.assertIsInstance(graph.G, torch.nn.Parameter)
        self.assertEqual(graph.G.shape, (config.N_AGENTS, config.N_AGENTS, config.K))

        diagonal = torch.stack(
            [graph.G[i, i].detach() for i in range(config.N_AGENTS)]
        )
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))

        with torch.no_grad():
            graph.G[0, 1, 0] = config.G_CLIP + 10.0

        edge = graph.get_edge(0, 1)
        self.assertLessEqual(float(edge[0]), config.G_CLIP)
        self.assertGreater(float(graph.G[0, 1, 0]), config.G_CLIP)

        fresh_graph = RelationGraph(init_std=0.01)
        loss = fresh_graph.get_edge(0, 1).sum()
        loss.backward()
        edge_grad = fresh_graph.G.grad[0, 1].abs().sum()
        self.assertGreater(float(edge_grad), 0.0)

    def test_ultimatum_adapter_keeps_outcome_and_features_domain_specific(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)

        snapshot = adapter.initial_runtime_snapshot()
        action_event = adapter.ground_action(0.7, snapshot=snapshot)
        transition = adapter.transition_event(
            snapshot, action_event, WorldResponse("accept"))
        outcome = adapter.outcome_from_transition(transition)
        self.assertEqual(outcome.response, "accept")
        self.assertTrue(outcome.terminal)
        self.assertFalse(hasattr(outcome, "payoff_A"))
        self.assertFalse(hasattr(outcome, "payoff_B"))
        self.assertFalse(hasattr(outcome, "pie_after"))
        self.assertAlmostEqual(
            outcome.payoff_for(adapter.focal_actor, adapter=adapter), 0.7
        )
        self.assertAlmostEqual(
            outcome.payoff_for(adapter.counterpart_actor, adapter=adapter), 0.3
        )

        features = adapter.outcome_features(outcome, role=adapter.focal_actor)
        self.assertEqual(features.shape, (len(adapter.outcome_feature_names),))
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
