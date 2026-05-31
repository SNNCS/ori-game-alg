"""End-to-end demo / smoke test for the original three-structure architecture.

Runs the full v5 dynamic loop on the ultimatum game and takes one optimizer
step, proving that gradients flow through all three structures:

    G  RelationGraph        -> grad present
    I  InterpretationEngine -> grad present
    T  FutureTreeGen.policy -> grad present

Run:  python demo.py
"""

import torch

import config
from agent import CognitiveAgent
from interpretation import build_context


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

    history = []
    for step in range(5):
        # Proposer A makes an offer (keeps `bid`, responder gets 1-bid).
        bid = 0.6 + 0.05 * step
        ctx = build_context(turn_idx=step, session_len=8,
                            prev_reject_rate=0.2, status_gap=0.1)

        out = agent.interpret_and_plan(
            bid, actor_i=config.ACTOR_A, context=ctx, history=history)
        Z, root, metrics = out["Z"], out["tree"], out["metrics"]

        # Loss = cognitive dissonance + rule regulariser - tree value (reward).
        L_diss = agent.dissonance(out["s"], Z)
        L_reg = agent.rule_reg()
        reward = metrics["path_quality"]
        loss = L_diss + 0.01 * L_reg - reward

        opt.zero_grad()
        loss.backward()

        gG = grad_norm(agent.G)
        gI = grad_norm(agent.interp)
        gT = grad_norm(agent.tree.policy)
        opt.step()

        print(f"step {step}: bid={bid:.2f}  loss={loss.item():+.4f}  "
              f"diss={L_diss.item():.4f}  reward={reward.item():.4f}")
        print(f"         metrics: optionality={metrics['optionality'].item():.3f}  "
              f"risk_floor={metrics['risk_floor'].item():.3f}  "
              f"path_quality={metrics['path_quality'].item():.3f}")
        print(f"         grad norms  G={gG:.4e}  I={gI:.4e}  T.policy={gT:.4e}  "
              f"edge_var={agent.G.edge_variance():.4e}")

        # Record the most likely response at the root for path dependence.
        z_B = Z[config.ACTOR_B]
        probs = agent.tree.policy(z_B, bid,
                                  tolerance=agent.tolerance(z_B)).detach()
        resp = config.RESPONSES[int(torch.argmax(probs))]
        history.append({"response": resp})
        agent.update_history(out["s"], resp, Z)
        print(f"         responder most-likely: {resp}  "
              f"P(accept/reject/counter)={[round(float(x),3) for x in probs]}")
        print("-" * 60)

    # Assert the three structures all received gradient at least once.
    assert gG > 0, "RelationGraph (G) got no gradient!"
    assert gI > 0, "InterpretationEngine (I) got no gradient!"
    assert gT > 0, "FutureTreeGen.policy (T) got no gradient!"
    print("OK: gradients flowed through all three structures (G, I, T).")
    print(f"final G[A,B] vs G[B,A] asymmetry: "
          f"{agent.G.asymmetry(config.ACTOR_A, config.ACTOR_B):.4f}")


if __name__ == "__main__":
    main()
