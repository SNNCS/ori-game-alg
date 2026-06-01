import unittest

import torch

import config
from agent import CognitiveAgent
from generic_adapter import GenericGameAdapter
from game_spec import (
    AddPayoff, ControlSpec, EntitySpec, FeatureSpec, GameSpec, GroundedAction,
    ResponseSpec, RoleBinding, SetTerminal, StateVarSpec, TransitionSpec,
    control, payoff,
)
from interpretation import build_context
from specs import BENCHMARK_SPECS, PUBLIC_GOODS_SPEC, ULTIMATUM_SPEC


class GenericGameSpecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(config.SEED)

    def test_benchmark_specs_instantiate_agent_and_act(self):
        context = build_context(turn_idx=0, session_len=8)
        for spec in BENCHMARK_SPECS:
            with self.subTest(spec=spec.name):
                adapter = GenericGameAdapter(spec)
                agent = CognitiveAgent(adapter=adapter)
                out = agent.act(context=context)
                decision = out["decision"]

                self.assertIsInstance(decision.selected.action, GroundedAction)
                self.assertEqual(
                    len(decision.futures),
                    adapter.action_affordance().n_candidates,
                )
                self.assertEqual(
                    len(decision.selected_future.tree.children),
                    len(adapter.response_labels()),
                )
                self.assertTrue(torch.isfinite(decision.expected_utility))

    def test_decode_action_supports_continuous_binary_and_categorical_controls(self):
        spec = GameSpec(
            name="mixed_controls",
            entities=(EntitySpec("a"), EntitySpec("b")),
            roles=RoleBinding("a", "b"),
            state_vars=(StateVarSpec("paths_open", init=0.0),),
            action_controls=(
                ControlSpec("amount", low=-1.0, high=1.0),
                ControlSpec("switch", kind="binary", low=0.0, high=1.0),
                ControlSpec(
                    "mode",
                    kind="categorical",
                    categories=("x", "y", "z"),
                ),
            ),
            responses=(ResponseSpec("done"),),
            transitions=(TransitionSpec("done", (SetTerminal(True),)),),
            outcome_features=(FeatureSpec("self_payoff", payoff("focal")),),
        )
        action = GenericGameAdapter(spec).decode_action(torch.zeros(5))

        self.assertIsInstance(action, GroundedAction)
        self.assertIn("amount", action.controls)
        self.assertIn("switch", action.controls)
        self.assertIn("mode", action.controls)
        self.assertEqual(action.controls["mode"].shape, (3,))
        self.assertAlmostEqual(float(action.controls["mode"].sum()), 1.0, places=5)

    def test_dsl_effects_update_payoffs_and_terminal_state(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        action = adapter.decode_action(torch.tensor([0.0]))
        state = adapter.initial_tree_state()

        next_state = adapter.transition(state, action, "accept")

        self.assertTrue(next_state["terminal"])
        self.assertAlmostEqual(
            float(next_state["payoffs"][adapter.focal_actor]),
            float(action.primary_value),
            places=5,
        )
        self.assertAlmostEqual(
            float(next_state["payoffs"][adapter.counterpart_actor]),
            1.0 - float(action.primary_value),
            places=5,
        )

    def test_continuous_control_keeps_gradient_through_outcome_quality(self):
        adapter = GenericGameAdapter(ULTIMATUM_SPEC)
        latent = torch.tensor([0.25], requires_grad=True)
        action = adapter.decode_action(latent)
        state = adapter.initial_tree_state()
        next_state = adapter.transition(state, action, "accept")

        quality = adapter.outcome_quality(next_state, adapter.focal_actor)
        quality.backward()

        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)

    def test_public_goods_aggregate_event_runs_future_tree(self):
        adapter = GenericGameAdapter(PUBLIC_GOODS_SPEC)
        agent = CognitiveAgent(adapter=adapter)

        out = agent.act(context=build_context(turn_idx=0, session_len=8))
        tree = out["decision"].selected_future.tree

        self.assertEqual(len(tree.children), len(adapter.response_labels()))
        self.assertTrue(torch.isfinite(out["decision"].expected_utility))


if __name__ == "__main__":
    unittest.main()
