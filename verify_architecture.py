"""Verification for the first-principles closed-loop architecture.

This is intentionally small and executable without a test framework:

    python -B verify_architecture.py

It checks that:
  * the public snapshot -> act -> transition -> trajectory -> learn loop works;
  * outgoing signals are generated as learned vectors;
  * ablating understanding changes the action-value landscape.
"""

import torch

import config
from agent import CognitiveAgent
from generic_adapter import GenericGameAdapter
from interpretation import build_context
from runtime import WorldResponse
from specs.ultimatum import build_ultimatum_spec
from trajectory import Trajectory


def main():
    torch.manual_seed(config.SEED)
    agent = CognitiveAgent()
    ctx = build_context(turn_idx=0, session_len=8,
                        prev_reject_rate=0.2, status_gap=0.1)
    snapshot = agent.runtime_snapshot()

    action_event = agent.act(snapshot=snapshot, context=ctx)
    decision = agent._last_decision
    signal_vec = decision.selected.signal.vector
    assert signal_vec.shape == (config.SIGNAL_DIM,), signal_vec.shape
    latent = decision.selected.latent_action
    assert latent is not None
    assert latent.shape == (config.ACTION_LATENT_DIM,), latent.shape
    assert len(agent._last_generated_interventions.candidates) == config.N_GENERATED_ACTIONS

    observed_response = agent.adapter.response_labels()[0]
    transition = agent.transition_runtime(
        snapshot, action_event, WorldResponse(observed_response))
    step = agent.build_trajectory_step(decision, action_event, transition)
    agent.learn(Trajectory((step,), transition.terminal_outcome))
    experience = agent._last_experience
    assert len(agent.observed_history) == 1
    assert experience.outcome.response == observed_response

    report = agent.probe_understanding_usefulness(
        context=ctx, snapshot=agent.runtime_snapshot())
    assert abs(float(report.utility_delta)) > 1e-6, report
    assert float(report.score_delta_norm) > 1e-6, report
    assert float(report.action_prob_delta_norm) > 1e-6, report

    variant_adapter = GenericGameAdapter(build_ultimatum_spec(
        responses=("accept", "reject", "counter", "delay")))
    variant_agent = CognitiveAgent(adapter=variant_adapter)
    variant_agent.act(snapshot=variant_agent.runtime_snapshot(), context=ctx)
    variant_decision = variant_agent._last_decision
    assert variant_agent.tree.policy.n_responses == 4
    assert len(variant_decision.selected_future.tree.children) == 4
    assert variant_agent.history.W_enc.in_features == config.M + 4 + config.D

    print("OK: public runtime action/transition/trajectory learning loop works.")
    print("OK: latent actions are generated and adapter-decoded.")
    print("OK: selected intervention carries a learned outgoing signal.")
    print("OK: ablating understanding changes expected position and policy scores.")
    print("OK: adapter response count drives history and branch-policy sizes.")
    print(f"    full_action={report.full_action:.2f}  "
          f"ablated_action={report.ablated_action:.2f}  "
          f"utility_delta={float(report.utility_delta):+.6f}  "
          f"score_delta_norm={float(report.score_delta_norm):.6f}")


if __name__ == "__main__":
    main()
