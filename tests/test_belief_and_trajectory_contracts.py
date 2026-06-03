import unittest

import torch

import config
from agent import CognitiveAgent
from runtime import WorldResponse
from trajectory import Trajectory, TrajectoryStep


class BeliefAndTrajectoryContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(config.SEED)

    def test_agent_builds_belief_state_from_runtime_observation(self):
        agent = CognitiveAgent()
        snapshot = agent.runtime_snapshot()
        observation = agent.observe_runtime(snapshot)
        belief = agent.update_belief(observation)

        self.assertEqual(belief.observation, observation)
        self.assertEqual(belief.observation_embedding.shape, (config.K_DIM,))
        self.assertIsNone(belief.Z)
        self.assertIsNone(belief.action_signal)
        self.assertIn(agent.adapter.focal_actor, belief.sigma)

    def test_trajectory_learning_uses_world_response_target(self):
        agent = CognitiveAgent()
        snapshot = agent.runtime_snapshot()
        action_event = agent.act(snapshot=snapshot)
        decision = agent._last_decision
        response = WorldResponse(agent.adapter.response_labels()[0])
        transition = agent.transition_runtime(snapshot, action_event, response)
        outcome = agent.adapter.outcome_from_transition(transition)
        experience = agent.build_experience(decision, outcome)

        trajectory = Trajectory((
            TrajectoryStep(
                decision=decision,
                action_event=action_event,
                world_response=response,
                transition=transition,
                outcome=outcome,
                realized_utility=experience.realized_utility,
            ),
        ), terminal_outcome=transition.terminal_outcome)
        signal = agent.learning.build_trajectory_signal(
            trajectory, role=agent.adapter.focal_actor)
        predicted_value = decision.scores[decision.selected_index]
        terminal_value = torch.as_tensor(
            transition.terminal_outcome.payoff_for(agent.adapter.focal_actor),
            device=predicted_value.device,
            dtype=predicted_value.dtype,
        )
        expected_value_loss = torch.nn.functional.mse_loss(
            predicted_value, terminal_value.detach())

        self.assertTrue(torch.isfinite(signal.total_loss))
        self.assertTrue(torch.allclose(signal.value_loss, expected_value_loss))
        self.assertEqual(outcome.world_response.label, response.label)

    def test_terminal_return_is_assigned_to_every_trajectory_step(self):
        agent = CognitiveAgent()
        snapshot0 = agent.runtime_snapshot()

        action0 = agent.act(snapshot=snapshot0)
        decision0 = agent._last_decision
        response0 = WorldResponse("counter")
        transition0 = agent.transition_runtime(snapshot0, action0, response0)
        step0 = agent.build_trajectory_step(decision0, action0, transition0)

        snapshot1 = transition0.after
        action1 = agent.act(snapshot=snapshot1)
        decision1 = agent._last_decision
        response1 = WorldResponse("accept")
        transition1 = agent.transition_runtime(snapshot1, action1, response1)
        step1 = agent.build_trajectory_step(decision1, action1, transition1)

        trajectory = Trajectory(
            (step0, step1),
            terminal_outcome=transition1.terminal_outcome,
        )
        signal = agent.learning.build_trajectory_signal(
            trajectory, role=agent.adapter.focal_actor)

        terminal_value = torch.as_tensor(
            transition1.terminal_outcome.payoff_for(agent.adapter.focal_actor),
            device=decision1.scores.device,
            dtype=decision1.scores.dtype,
        )
        expected = torch.stack([
            torch.nn.functional.mse_loss(
                decision0.scores[decision0.selected_index],
                terminal_value.detach()),
            torch.nn.functional.mse_loss(
                decision1.scores[decision1.selected_index],
                terminal_value.detach()),
        ]).mean()

        self.assertTrue(torch.allclose(signal.value_loss, expected))

    def test_terminal_return_resolves_role_alias_through_adapter(self):
        agent = CognitiveAgent()
        snapshot = agent.runtime_snapshot()
        action_event = agent.act(snapshot=snapshot)
        decision = agent._last_decision
        response = WorldResponse("accept")
        transition = agent.transition_runtime(snapshot, action_event, response)
        step = agent.build_trajectory_step(decision, action_event, transition)
        trajectory = Trajectory((step,), terminal_outcome=transition.terminal_outcome)

        signal_by_id = agent.learning.build_trajectory_signal(
            trajectory, role=agent.adapter.focal_actor, adapter=agent.adapter)
        signal_by_alias = agent.learning.build_trajectory_signal(
            trajectory, role="focal", adapter=agent.adapter)

        self.assertTrue(torch.allclose(
            signal_by_alias.value_loss,
            signal_by_id.value_loss,
        ))

    def test_nonterminal_trajectory_does_not_use_realized_utility_as_target(self):
        agent = CognitiveAgent()
        snapshot = agent.runtime_snapshot()
        action_event = agent.act(snapshot=snapshot)
        decision = agent._last_decision
        response = WorldResponse("counter")
        transition = agent.transition_runtime(snapshot, action_event, response)
        step = agent.build_trajectory_step(decision, action_event, transition)
        trajectory = Trajectory((step,), terminal_outcome=None)

        signal = agent.learning.build_trajectory_signal(
            trajectory, role=agent.adapter.focal_actor)

        self.assertTrue(torch.allclose(
            signal.value_loss,
            decision.scores.new_zeros(()),
        ))

    def test_dissonance_default_skips_actor_not_counterpart(self):
        agent = CognitiveAgent()
        snapshot = agent.runtime_snapshot()
        agent.act(snapshot=snapshot)
        future = agent._last_decision.selected_future

        default_loss = agent.dissonance(future.signal_vec, future.Z)
        focal_skipped = agent.dissonance(
            future.signal_vec, future.Z,
            observed_actor=agent.adapter.focal_actor)
        counterpart_skipped = agent.dissonance(
            future.signal_vec, future.Z,
            observed_actor=agent.adapter.counterpart_actor)

        self.assertTrue(torch.allclose(default_loss, focal_skipped))
        self.assertGreater(float(default_loss), float(counterpart_skipped))


if __name__ == "__main__":
    unittest.main()
