import unittest

import torch

import config
from agent import CognitiveAgent
from interpretation import build_context
from runtime import WorldResponse
from trajectory import Trajectory


def grad_norm(module):
    grads = [p.grad.norm() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack(grads).norm())


class ClosedLoopRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(config.SEED)

    def test_public_act_observe_learn_loop_updates_detached_episode_state(self):
        agent = CognitiveAgent()
        context = build_context(turn_idx=0, session_len=8)
        snapshot = agent.runtime_snapshot()

        action_event = agent.act(snapshot=snapshot, context=context)
        decision = agent._last_decision

        self.assertEqual(decision.selected.signal.vector.shape, (config.SIGNAL_DIM,))
        self.assertEqual(decision.selected.latent_action.shape, (config.ACTION_LATENT_DIM,))

        response = WorldResponse(agent.adapter.response_labels()[0])
        transition = agent.transition_runtime(snapshot, action_event, response)
        step = agent.build_trajectory_step(decision, action_event, transition)
        signal = agent.learn(Trajectory((step,), transition.terminal_outcome))
        experience = agent._last_experience
        outcome = experience.outcome

        self.assertEqual(len(agent.observed_history), 1)
        self.assertEqual(experience.outcome.world_response.label, response.label)
        self.assertTrue(torch.isfinite(experience.realized_utility.value))
        self.assertTrue(torch.isfinite(signal.total_loss))

        for h_j in agent.h:
            self.assertFalse(h_j.requires_grad)

        focal = agent.adapter.focal_actor
        self.assertAlmostEqual(
            float(agent.omega[focal][0]),
            outcome.payoff_for(focal, adapter=agent.adapter),
            places=5,
        )

    def test_training_loss_backpropagates_through_closed_loop_modules(self):
        agent = CognitiveAgent()
        context = build_context(
            turn_idx=0,
            session_len=8,
            prev_reject_rate=0.2,
            status_gap=0.1,
        )

        snapshot = agent.runtime_snapshot()
        action_event = agent.act(snapshot=snapshot, context=context)
        decision = agent._last_decision
        future = decision.selected_future
        root_probs = torch.stack([child.prob for child in future.tree.children]).detach()
        response = WorldResponse(
            agent.adapter.response_labels()[int(torch.argmax(root_probs))])

        transition = agent.transition_runtime(snapshot, action_event, response)
        step = agent.build_trajectory_step(decision, action_event, transition)
        learning_signal = agent.learn(Trajectory((step,), transition.terminal_outcome))

        loss = (
            agent.dissonance(future.signal_vec, future.Z)
            + 0.01 * agent.rule_reg()
            + learning_signal.total_loss
        )

        agent.zero_grad(set_to_none=True)
        loss.backward()

        modules = {
            "G": agent.G,
            "I": agent.interp,
            "T.policy": agent.tree.policy,
            "action_gen": agent.action_gen,
            "signal_gen": agent.signal_gen,
            "future_utility": agent.utility,
        }
        for name, module in modules.items():
            self.assertGreater(grad_norm(module), 0.0, name)

    def test_understanding_ablation_changes_action_value_landscape(self):
        agent = CognitiveAgent()
        context = build_context(
            turn_idx=0,
            session_len=8,
            prev_reject_rate=0.2,
            status_gap=0.1,
        )

        report = agent.probe_understanding_usefulness(
            context=context, snapshot=agent.runtime_snapshot())

        self.assertGreater(abs(float(report.utility_delta)), 1e-6)
        self.assertGreater(float(report.score_delta_norm), 1e-6)
        self.assertGreater(float(report.action_prob_delta_norm), 1e-6)


if __name__ == "__main__":
    unittest.main()
