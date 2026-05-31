"""Structure ②  未来结果树  T = (N, B, P).

Spec (v5, section 三):
    T = (N, B, P)
    N = state nodes (each a situation vector in R^d)
    B = branches (action, response, prob, new-node)
    P = branch probabilities, *driven by the interpretation matrix Z*
    T_{t+1} = Reconstruct(Z_t, G_t, H)        -- the tree is generated, not fixed

The v5 reference computed branch probability from a hard-coded slice:
    dignity = z_B[4:12].mean()
    P(reject) = dignity*0.8 + (action-0.5)*0.4   ...

We re-implement T and remove that slicing (user directive: z is not sliced).
Branch probabilities now come from a small *learnable* policy over the whole
responder intent vector z_B plus the offer (un)fairness, and (optionally) the
responder's tolerance head. Probabilities stay as torch tensors so gradients
flow from the tree's value back into the policy, the interpretation engine, and
the relation graph -- this is the "kept optimization" for T.

Metrics (v5):
    optionality   -- here a differentiable normalized entropy of leaf mass
    risk_floor    -- min leaf quality
    path_quality  -- expected leaf quality
    apply_path_dep-- P(b|H) ∝ P(b) * exp(lambda * consistency(b, H))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Branch policy:  P(response | z_B, action)   -- learnable, no z slicing
# ---------------------------------------------------------------------------

class BranchPolicy(nn.Module):
    """Maps the responder's intent vector + offer fairness to a distribution
    over (accept, reject, counter).

    Replaces the v5 hard-coded `dignity = z_B[4:12].mean()` rule. The whole
    z_B is consumed by W_resp; the scalar (action-0.5) fairness gap is the
    only structural prior we keep (an unfairer grab should push toward reject),
    and even its effect is learned through w_action.
    """

    def __init__(self, d=config.D, use_tolerance=True):
        super().__init__()
        self.use_tolerance = use_tolerance
        self.W_resp = nn.Linear(d, 3, bias=True)        # intent -> response logits
        self.w_action = nn.Linear(1, 3, bias=False)     # fairness-gap modulation
        if use_tolerance:
            self.w_tol = nn.Linear(1, 3, bias=False)    # tolerance modulation

    def forward(self, z_B, action, tolerance=None):
        gap = torch.tensor([float(action) - 0.5],
                           device=z_B.device, dtype=z_B.dtype)
        logits = self.W_resp(z_B) + self.w_action(gap)
        if self.use_tolerance and tolerance is not None:
            tol = tolerance.reshape(1).to(z_B.dtype)
            logits = logits + self.w_tol(tol)
        return F.softmax(logits, dim=-1)                # (3,) accept/reject/counter


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("state", "action", "response", "prob", "children")

    def __init__(self, state, action=None, response=None, prob=None):
        self.state = state
        self.action = action
        self.response = response
        self.prob = prob              # torch scalar (keeps the graph alive)
        self.children = []


# ---------------------------------------------------------------------------
# FutureTreeGen
# ---------------------------------------------------------------------------

class FutureTreeGen(nn.Module):
    BIDS = config.BIDS
    RESPONSES = config.RESPONSES

    def __init__(self, rule, d=config.D, tolerance_head=None):
        super().__init__()
        self.rule = rule
        self.d = d
        self.tolerance_head = tolerance_head      # optional ToleranceHead
        self.policy = BranchPolicy(d=d, use_tolerance=tolerance_head is not None)

    # ----- branch probabilities (driven by Z) ------------------------------

    def _branch_probs(self, z_B, action):
        tol = None
        if self.tolerance_head is not None:
            tol = self.tolerance_head(z_B)
        return self.policy(z_B, action, tolerance=tol)

    # ----- transition ------------------------------------------------------

    def transition(self, state, action, response):
        new = dict(state)
        pie = state["pie"]
        if response == "accept":
            new["payoff_A"] = state["payoff_A"] + self.rule.compute_payoff(
                action, "accept", config.ACTOR_A, pie)
            new["payoff_B"] = state["payoff_B"] + self.rule.compute_payoff(
                action, "accept", config.ACTOR_B, pie)
            new["paths_open"] = 0.0                      # deal closed
        elif response == "reject":
            new["payoff_A"] = state["payoff_A"] + self.rule.outside
            new["payoff_B"] = state["payoff_B"] + self.rule.outside
            new["paths_open"] = 0.0                      # game over (spite)
        else:                                            # counter -> continue
            new["pie"] = pie * config.COUNTER_DISCOUNT   # the stake shrinks
            new["paths_open"] = max(0.0, state["paths_open"] - config.PATHS_OPEN_DECAY)
        return new

    # ----- intent drift across depth --------------------------------------

    @staticmethod
    def update_Z(Z, response):
        """Small whole-row perturbation of the responder's intent across a
        continue branch (v5 update_Z). No slicing -- the entire z_B row drifts.
        """
        Z2 = Z.clone()
        shift = 0.05 if response == "reject" else (-0.05 if response == "accept" else 0.0)
        Z2[config.ACTOR_B, :] = torch.tanh(Z2[config.ACTOR_B, :] + shift)
        return Z2

    # ----- tree construction ----------------------------------------------

    def generate(self, Z, sigma_root=None, depth=config.DEPTH):
        root_state = {"payoff_A": 0.0, "payoff_B": 0.0,
                      "pie": 1.0, "paths_open": 1.0, "sigma": sigma_root}
        return self._build(Z, root_state, depth, path_prob=None)

    def _build(self, Z, state, depth, path_prob):
        root = Node(state, prob=path_prob)
        z_B = Z[config.ACTOR_B]
        n_bids = len(self.BIDS)
        for bid in self.BIDS:
            probs = self._branch_probs(z_B, bid)          # (3,)
            for ri, resp in enumerate(self.RESPONSES):
                p = probs[ri] / n_bids                     # uniform prior over bids
                joint = p if path_prob is None else path_prob * p
                new_state = self.transition(state, bid, resp)
                child = Node(new_state, action=bid, response=resp, prob=joint)
                root.children.append(child)
                if depth > 1 and resp == "counter":
                    new_Z = self.update_Z(Z, resp)
                    sub = self._build(new_Z, new_state, depth - 1, joint)
                    child.children = sub.children
        return root

    # ----- evaluation ------------------------------------------------------

    @staticmethod
    def _quality(state, role=config.ACTOR_A):
        payoff = state["payoff_A"] if role == config.ACTOR_A else state["payoff_B"]
        return payoff + 0.2 * state["paths_open"]

    def _collect_leaves(self, node, leaves):
        if not node.children:
            leaves.append(node)
            return
        for c in node.children:
            self._collect_leaves(c, leaves)

    def evaluate(self, root, role=config.ACTOR_A):
        """Returns a dict of torch scalars (differentiable w.r.t. the policy,
        interpretation engine, and relation graph through the leaf probs).
        """
        leaves = []
        self._collect_leaves(root, leaves)
        P = torch.stack([leaf.prob for leaf in leaves])
        P = P / (P.sum() + 1e-8)
        Q = torch.tensor([self._quality(leaf.state, role) for leaf in leaves],
                         dtype=P.dtype, device=P.device)
        H = -(P * torch.log(P + 1e-12)).sum()
        H_max = torch.log(torch.tensor(float(max(len(P), 1)),
                                       dtype=P.dtype, device=P.device)) + 1e-12
        return {
            "optionality":  H / H_max,                 # normalized entropy in [0,1]
            "risk_floor":   Q.min(),
            "path_quality": (P * Q).sum(),
        }

    # ----- path dependence -------------------------------------------------

    @staticmethod
    def _scale_descendant_probs(node, scale):
        for child in node.children:
            child.prob = child.prob * scale
            FutureTreeGen._scale_descendant_probs(child, scale)

    def apply_path_dep(self, root, history, lam=config.LAMBDA):
        """P(b|H) ∝ P(b) * exp(lambda * consistency(b, H)) over root branches.

        Node probabilities are stored as joint path probabilities. When a root
        branch is reweighted, every descendant path under that branch must be
        scaled by the same factor or counter subtrees keep the stale mass.
        """
        if not history:
            return root
        recent = history[-10:]
        outs = [h.get("response", "counter") for h in recent]
        n = max(len(outs), 1)
        rate = {r: outs.count(r) / n for r in self.RESPONSES}

        def consistency(resp):
            return rate.get(resp, 0.0)

        new_probs = []
        for c in root.children:
            factor = torch.exp(torch.tensor(lam * consistency(c.response),
                                            dtype=c.prob.dtype, device=c.prob.device))
            new_probs.append(c.prob * factor)
        total = torch.stack(new_probs).sum() + 1e-8
        for c, np_ in zip(root.children, new_probs):
            old_prob = c.prob
            adjusted = np_ / total
            scale = adjusted / (old_prob + 1e-8)
            c.prob = adjusted
            self._scale_descendant_probs(c, scale)
        return root
