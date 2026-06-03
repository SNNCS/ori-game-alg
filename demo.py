"""End-to-end demo / smoke test for the first closed-loop architecture.

Runs the dynamic loop on the ultimatum game and takes optimizer steps, proving
that gradients flow through interpretation, future simulation, utility scoring,
and action selection:

    G  RelationGraph        -> grad present
    I  InterpretationEngine -> grad present
    T  FutureTreeGen.policy -> grad present
    A  ActionGenerator      -> grad present
    S  SignalGenerator      -> grad present
    U  FuturePosition       -> grad present

Run:  python demo.py
"""

import torch

import config
from agent import CognitiveAgent
from interpretation import build_context
from runtime import WorldResponse
from trajectory import Trajectory


def grad_norm(module):
    g = [p.grad.norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(g).norm().item()) if g else 0.0


def main():
    torch.manual_seed(config.SEED)
    agent = CognitiveAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=config.LR)

    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"trainable params: {n_params}")
    print(f"INPUT_DIM={config.INPUT_DIM}  D={config.D}  K={config.K}  P={config.P}")
    print("-" * 60)
    response_labels = agent.adapter.response_labels()

    for step in range(5):
        ctx = build_context(turn_idx=step, session_len=8,
                            prev_reject_rate=0.2, status_gap=0.1)
        snapshot = agent.runtime_snapshot()

        # The proposer now chooses its own bid by simulating candidate futures.
        action_event = agent.act(snapshot=snapshot, context=ctx)
        decision = agent._last_decision
        future = decision.selected_future
        bid = decision.selected.action
        bid_value = float(bid)
        Z, metrics = future.Z, future.metrics

        # Observe an outcome from the response model. This remains a raw
        # outcome until the utility interface evaluates it.
        root_probs = torch.stack(
            [child.prob for child in future.tree.children]).detach()
        resp = response_labels[int(torch.argmax(root_probs))]
        transition = agent.transition_runtime(
            snapshot, action_event, WorldResponse(resp))
        step_record = agent.build_trajectory_step(
            decision, action_event, transition)
        learning_signal = agent.learn(
            Trajectory((step_record,), transition.terminal_outcome))
        experience = agent._last_experience
        outcome = experience.outcome
        focal_payoff = outcome.payoff_for(
            agent.adapter.focal_actor, adapter=agent.adapter)

        # Outcome is not reward: total learning separates prediction, value,
        # policy, interpretation, and rule-regularization terms.
        L_diss = agent.dissonance(future.signal_vec, Z)
        L_reg = agent.rule_reg()
        expected_position = decision.expected_utility
        L_exp = learning_signal.total_loss
        loss = L_diss + 0.01 * L_reg + L_exp

        opt.zero_grad()
        loss.backward()

        gG = grad_norm(agent.G)
        gI = grad_norm(agent.interp)
        gT = grad_norm(agent.tree.policy)
        gA = grad_norm(agent.action_gen)
        gS = grad_norm(agent.signal_gen)
        gU = grad_norm(agent.utility)
        opt.step()

        print(f"step {step}: bid={bid_value:.2f}  loss={loss.item():+.4f}  "
              f"diss={L_diss.item():.4f}  "
              f"expected_position={expected_position.item():.4f}")
        print(f"         outcome: response={outcome.response}  "
              f"focal_payoff={focal_payoff:.3f}  "
              f"realized_utility={experience.realized_utility.value.item():.3f}")
        print(f"         learning: pred={experience.learning_signal.prediction_loss.item():.4f}  "
              f"value={experience.learning_signal.value_loss.item():.4f}  "
              f"policy={experience.learning_signal.policy_loss.item():+.4f}")
        print(f"         metrics: optionality={metrics['optionality'].item():.3f}  "
              f"risk_floor={metrics['risk_floor'].item():.3f}  "
              f"path_quality={metrics['path_quality'].item():.3f}")
        print(f"         grad norms  G={gG:.4e}  I={gI:.4e}  "
              f"T.policy={gT:.4e}  A={gA:.4e}  S={gS:.4e}  U={gU:.4e}  "
              f"edge_var={agent.G.edge_variance():.4e}")
        if decision.selected.latent_action is not None:
            print(f"         latent_action_norm="
                  f"{decision.selected.latent_action.detach().norm().item():.3f}")
        print(f"         outgoing_signal_norm="
              f"{future.outgoing_signal.vector.detach().norm().item():.3f}")
        print(f"         candidate scores="
              f"{[round(float(x),3) for x in decision.scores.detach()]}  "
              f"action_probs="
              f"{[round(float(x),3) for x in decision.action_probs.detach()]}")

        probs = root_probs / (root_probs.sum() + 1e-8)
        print(f"         responder most-likely: {resp}  "
              f"P({list(response_labels)})="
              f"{[round(float(x),3) for x in probs]}")
        print("-" * 60)

    # Assert the three structures all received gradient at least once.
    assert gG > 0, "RelationGraph (G) got no gradient!"
    assert gI > 0, "InterpretationEngine (I) got no gradient!"
    assert gT > 0, "FutureTreeGen.policy (T) got no gradient!"
    assert gA > 0, "ActionGenerator (A) got no gradient!"
    assert gS > 0, "SignalGenerator (S) got no gradient!"
    assert gU > 0, "FuturePositionEvaluator (U) got no gradient!"
    print("OK: agent generated actions/signals and gradients flowed through G, I, T, A, S, U.")
    print(f"final G[A,B] vs G[B,A] asymmetry: "
          f"{agent.G.asymmetry(agent.adapter.focal_actor, agent.adapter.counterpart_actor):.4f}")


if __name__ == "__main__":
    main()
