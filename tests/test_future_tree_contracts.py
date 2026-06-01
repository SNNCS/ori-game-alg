import unittest

import torch

import config
from agent import CognitiveAgent
from interpretation import build_context


def collect_leaves(node):
    if not node.children:
        return [node]
    leaves = []
    for child in node.children:
        leaves.extend(collect_leaves(child))
    return leaves


class FutureTreeContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(config.SEED)
        self.agent = CognitiveAgent()
        self.context = build_context(
            turn_idx=0,
            session_len=8,
            prev_reject_rate=0.2,
            status_gap=0.1,
        )

    def test_interpretation_produces_intent_matrix_and_outgoing_signal(self):
        out = self.agent.interpret_and_plan(0.7, context=self.context)

        Z = out["Z"]
        actor = self.agent.adapter.focal_actor
        counterpart = self.agent.adapter.counterpart_actor

        self.assertEqual(Z.shape, (self.agent.n, config.D))
        self.assertTrue(torch.allclose(Z[actor], torch.zeros_like(Z[actor])))
        self.assertGreater(float(Z[counterpart].abs().sum()), 0.0)
        self.assertEqual(out["z_C"].shape, (config.D,))
        self.assertEqual(out["outgoing_signal"].vector.shape, (config.SIGNAL_DIM,))

    def test_simulated_tree_metrics_are_finite_and_differentiable(self):
        out = self.agent.interpret_and_plan(0.7, context=self.context, depth=2)
        leaves = collect_leaves(out["tree"])
        leaf_mass = torch.stack([leaf.prob for leaf in leaves]).sum()

        self.assertTrue(torch.allclose(leaf_mass, torch.ones_like(leaf_mass), atol=1e-5))
        self.assertEqual(len(out["tree"].children), len(self.agent.adapter.response_labels()))

        metrics = out["metrics"]
        for name in ("optionality", "risk_floor", "path_quality"):
            self.assertEqual(metrics[name].shape, ())
            self.assertTrue(torch.isfinite(metrics[name]))

        self.assertTrue(metrics["path_quality"].requires_grad)

        loss = metrics["path_quality"] + metrics["optionality"]
        self.agent.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = sum(
            p.grad.abs().sum()
            for p in self.agent.tree.policy.parameters()
            if p.grad is not None
        )
        self.assertGreater(float(grad_norm), 0.0)

    def test_path_dependence_preserves_coherent_leaf_probability_mass(self):
        history = [
            {"response": "counter"},
            {"response": "counter"},
            {"response": "accept"},
        ]
        out = self.agent.interpret_and_plan(
            0.7,
            context=self.context,
            depth=2,
            history=history,
        )

        root_mass = torch.stack([child.prob for child in out["tree"].children]).sum()
        leaf_mass = torch.stack([leaf.prob for leaf in collect_leaves(out["tree"])]).sum()

        self.assertTrue(torch.allclose(root_mass, torch.ones_like(root_mass), atol=1e-5))
        self.assertTrue(torch.allclose(leaf_mass, torch.ones_like(leaf_mass), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
