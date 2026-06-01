"""Verification for the first-principles closed-loop architecture.

This is intentionally small and executable without a test framework:

    python -B verify_architecture.py

It checks that:
  * the public act -> observe -> learn loop works;
  * outgoing signals are generated as learned vectors;
  * ablating understanding changes the action-value landscape.
"""

import torch

import config
from agent import CognitiveAgent
from game_adapter import UltimatumGameAdapter
from game_rule import UltimatumRule
from interpretation import build_context


def main():
    torch.manual_seed(config.SEED)
    agent = CognitiveAgent()
    ctx = build_context(turn_idx=0, session_len=8,
                        prev_reject_rate=0.2, status_gap=0.1)

    acted = agent.act(context=ctx)
    decision = acted["decision"]
    signal_vec = decision.selected.signal.vector
    assert signal_vec.shape == (config.SIGNAL_DIM,), signal_vec.shape
    latent = decision.selected.latent_action
    assert latent is not None
    assert latent.shape == (config.ACTION_LATENT_DIM,), latent.shape
    assert len(acted["generated_interventions"].candidates) == config.N_GENERATED_ACTIONS

    observed_response = agent.adapter.response_labels()[0]
    outcome = agent.observe(observed_response, decision=decision)
    experience = agent.learn(outcome, decision=decision)
    assert len(agent.observed_history) == 1
    assert experience.outcome.response == observed_response

    report = agent.probe_understanding_usefulness(context=ctx)
    assert abs(float(report.utility_delta)) > 1e-6, report
    assert float(report.score_delta_norm) > 1e-6, report
    assert float(report.action_prob_delta_norm) > 1e-6, report

    variant_adapter = UltimatumGameAdapter(
        UltimatumRule(),
        responses=("accept", "reject", "counter", "delay"),
    )
    variant_agent = CognitiveAgent(adapter=variant_adapter)
    variant_decision = variant_agent.act(context=ctx)["decision"]
    assert variant_agent.tree.policy.n_responses == 4
    assert len(variant_decision.selected_future.tree.children) == 4
    assert variant_agent.history.W_enc.in_features == config.M + 4 + config.D

    print("OK: public act/observe/learn loop works.")
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
