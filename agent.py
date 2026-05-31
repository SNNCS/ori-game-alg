"""CognitiveAgent -- wires the three structures into one optimizable module.

    G  RelationGraph        (structure ①)
    I  InterpretationEngine (structure ③)  + r_j, BayesianInverse, ToleranceHead
    T  FutureTreeGen        (structure ②)

Plus the per-agent situation pieces (role embedding, history, resource, K).

Everything is an nn.Module, so a single optimizer over `self.parameters()`
trains G, I and T jointly via autograd -- the kept optimization. The agent
exposes one `interpret_and_plan` step that runs the full v5 dynamic loop:

    s = phi(a, S, K)
    Z = [ I_j(s, G[j,i,:], r_j, sigma_j) ]_j          # interpretation
    z_C = propagate(z_B, ...)                          # re-signalling B -> C
    T   = Reconstruct(Z)                               # future tree
    z*  = BayesianInverse(s, z_prior)                  # dissonance target
"""

import torch
import torch.nn as nn

import config
from relation_graph import RelationGraph
from interpretation import (
    InterpretationEngine, RuleInterpretation, BayesianInverse, ToleranceHead,
    build_signal, build_context, dissonance_loss,
)
from situation import (
    RoleEmbedding, HistorySummarizer, init_resource, update_resource,
    init_knowledge, build_sigma,
)
from game_rule import UltimatumRule
from future_tree import FutureTreeGen


class CognitiveAgent(nn.Module):
    def __init__(self, rule=None, n_agents=config.N_AGENTS):
        super().__init__()
        self.n = n_agents
        self.rule = rule if rule is not None else UltimatumRule()

        # structure ①
        self.G = RelationGraph(n_agents=n_agents)

        # structure ③ + heads
        self.interp = InterpretationEngine()
        self.bayes = BayesianInverse()
        self.tolerance = ToleranceHead()
        self.rules = nn.ModuleList([
            RuleInterpretation(init_r=self.rule.r_public) for _ in range(n_agents)
        ])

        # situation
        self.roles = RoleEmbedding()
        self.history = HistorySummarizer()

        # structure ②  (shares the tolerance head so the tree reads B's z too)
        self.tree = FutureTreeGen(self.rule, tolerance_head=self.tolerance)

        # mutable per-episode situation buffers (not parameters)
        self.h = [torch.zeros(config.H_DIM) for _ in range(n_agents)]
        self.omega = [init_resource(i, {0: 0.0, 1: 0.0, 2: 0.0})
                      for i in range(n_agents)]
        self.K = init_knowledge()

    # ----- situation assembly ---------------------------------------------

    def sigma_of(self, j):
        rho = self.roles(j)
        return build_sigma(rho, self.h[j], self.omega[j], self.K)

    def r_of(self, j):
        return self.rules[j].r_j

    # ----- one full interpretive step -------------------------------------

    def interpret_and_plan(self, bid, actor_i=config.ACTOR_A, context=None,
                           depth=config.DEPTH, history=None):
        """Run the v5 loop for a single proposer action (the bid).

        Returns a dict with Z, the propagated z_C, the future tree, and its
        evaluation metrics. Everything is differentiable.
        """
        s = build_signal(bid, context)
        r_dict = {j: self.r_of(j) for j in range(self.n)}
        sigma_dict = {j: self.sigma_of(j) for j in range(self.n)}

        Z = self.interp.compute_Z(s, actor_i, self.G, r_dict, sigma_dict)

        # B -> C re-signalling (B tells the observer about A's offer)
        z_C = self.interp.propagate(
            Z[config.ACTOR_B], self.G, config.ACTOR_B, config.ACTOR_C,
            r_dict[config.ACTOR_C], sigma_dict[config.ACTOR_C])

        root = self.tree.generate(Z, sigma_root=sigma_dict[actor_i], depth=depth)
        self.tree.apply_path_dep(root, history, lam=config.LAMBDA)
        metrics = self.tree.evaluate(root, role=actor_i)
        return {"s": s, "Z": Z, "z_C": z_C, "tree": root, "metrics": metrics}

    # ----- mutable episode history ----------------------------------------

    def update_history(self, signal_vec, response, Z, gamma=config.GAMMA):
        """Commit the observed response into each agent's history summary.

        `h` is episode state rather than a parameter. We detach the new summary
        so the next turn starts from the observation without backpropagating
        through previous optimizer steps.
        """
        for j in range(self.n):
            z_j = Z[j].detach()
            s_j = torch.as_tensor(
                signal_vec, dtype=z_j.dtype, device=z_j.device).detach()
            h_j = self.h[j].to(device=z_j.device, dtype=z_j.dtype)
            self.h[j] = self.history.update_h(
                h_j, s_j, response, z_j, gamma=gamma).detach()
        return self.h

    # ----- cognitive-dissonance loss --------------------------------------

    def dissonance(self, s, Z, observed_actor=config.ACTOR_B):
        """L = sum_j KL(z_j || z_j*), z_j* = BayesianInverse(s, z_j).

        The "real action" s corrects each observer's prior intent; the gap is
        the structural error the doc calls cognitive dissonance (认知失调).
        """
        total = Z.new_zeros(())
        for j in range(self.n):
            if j == observed_actor:
                continue
            z_j = Z[j]
            if torch.count_nonzero(z_j) == 0:
                continue
            z_star = self.bayes(s, z_j)
            total = total + dissonance_loss(z_j, z_star)
        return total

    def rule_reg(self):
        return sum(self.rules[j].regularization_loss(self.rule.r_public)
                   for j in range(self.n))
