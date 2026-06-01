"""Structure ③  解释机制  I_j(a, S, G, H)  ->  Z in R^(n x d).

Spec (v5, section 二):
    z_j = tanh(W_z [s || G[j,i,:]] + b_z)               # shared net, own edge
    Z[j,:] = z_j = I_j(a_i, S, G, H)                    # multi-observer matrix
    propagate: z_C = I_C(signal(z_B), G[C,B,:])

Kept optimizations / user directives:
    * The interpretation input keeps the rule-interpretation block r_j and the
      situation block sigma_j:   z = tanh(W_z[s || edge || r_j || sigma_j]).
    * z is NOT sliced into the 5 semantic segments. No code reads z[0:4] /
      z[4:12] / ... as fixed meanings; downstream heads consume the whole z.
    * BayesianInverse maps the *real action signal* (not utterance text) into an
      intent-correction vector z*, giving the cognitive-dissonance target
      L = sum_j KL(z_j || z_j*) without any discrete response-type classifier.

This module also hosts the small heads that live on top of z:
    RuleInterpretation (r_j), BayesianInverse (z*), ToleranceHead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Signal s = phi(a_i, S, K)   (the action IS the signal in the ultimatum game)
# ---------------------------------------------------------------------------

def build_context(turn_idx=0, session_len=0, prev_reject_rate=0.0,
                  status_gap=0.0, urgency=0.0):
    """Pack the N_CONTEXT context floats into a tensor (N_CONTEXT,)."""
    return torch.tensor([
        min(turn_idx / 20.0, 1.0),
        min(session_len / 50.0, 1.0),
        float(prev_reject_rate),
        float(status_gap),
        float(urgency),
    ], dtype=torch.float32)


def build_signal(bid, context=None):
    """s = [bid, 1-bid, fairness_dev] || context   ->  (M,)

    bid is the proposer's *kept* share in [0,1]. fairness_dev = 2*|bid-0.5|
    is 0 for an even split and 1 for a total grab. No learnable params: the
    signal is a deterministic encoding of the physical action + context.
    """
    if context is None:
        context = build_context()
    context = torch.as_tensor(context, dtype=torch.float32)
    bid = torch.as_tensor(bid, dtype=context.dtype, device=context.device)
    bid = bid.reshape(())
    head = torch.stack([
        bid,
        1.0 - bid,
        2.0 * torch.abs(bid - 0.5),
    ])
    return torch.cat([head, context.to(dtype=head.dtype, device=head.device)],
                     dim=-1)


# ---------------------------------------------------------------------------
# InterpretationEngine  (structure ③ core)
# ---------------------------------------------------------------------------

class InterpretationEngine(nn.Module):
    """z_j = tanh(W_z [s || edge || r_j || sigma_j])."""

    def __init__(self, d=config.D, input_dim=config.INPUT_DIM):
        super().__init__()
        self.d = d
        self.input_dim = input_dim
        self.W_z = nn.Linear(input_dim, d, bias=True)

    def infer_intent(self, s, edge, r_j, sigma_j):
        x = torch.cat([s, edge, r_j, sigma_j], dim=-1)
        assert x.shape[-1] == self.input_dim, (x.shape, self.input_dim)
        return torch.tanh(self.W_z(x))

    def compute_Z(self, s, actor_i, G, r_dict, sigma_dict):
        """Multi-observer interpretation matrix Z, shape (n, d).

        Every observer j != actor_i reads actor_i's action through *its own*
        relation edge G[j, actor_i, :]. The shared W_z is the common human
        interpretive faculty; the differing edges are the differing stances.
        Row actor_i is zeros (an agent does not interpret itself here).
        """
        Z = []
        for j in range(G.n):
            if j == actor_i:
                Z.append(torch.zeros(self.d, device=s.device, dtype=s.dtype))
            else:
                edge_ji = G.get_edge(j, actor_i)
                Z.append(self.infer_intent(s, edge_ji, r_dict[j], sigma_dict[j]))
        return torch.stack(Z, dim=0)

    def propagate(self, z_B, G, source_B, target_C, r_C, sigma_C):
        """Re-signalling B -> C:  z_C = I_C(signal(z_B), G[C,B,:], r_C, sigma_C).

        signal(z_B) = z_B[:M], the first M dims of B's intent reused as a
        compact propagated signal (v5 spec: prop_signal = z_B[:m]). r_C and
        sigma_C are kept so the propagated read still carries C's rule stance
        and situation.
        """
        prop_signal = z_B[:config.M]
        edge_CB = G.get_edge(target_C, source_B)
        return self.infer_intent(prop_signal, edge_CB, r_C, sigma_C)


# ---------------------------------------------------------------------------
# RuleInterpretation  r_j  (per-agent rule stance with diagonal regulariser)
# ---------------------------------------------------------------------------

class RuleInterpretation(nn.Module):
    def __init__(self, p=config.P, init_r=None):
        super().__init__()
        self.p = p
        if init_r is None:
            init_r = torch.zeros(p)
        else:
            init_r = torch.as_tensor(init_r, dtype=torch.float32).clone()
        self.r_j = nn.Parameter(init_r)
        # Lambda ~ exp(N(-2.5, 0.3)) ~ 0.08: weak, asymmetric pull to r_public.
        self.log_lambda = nn.Parameter(torch.randn(p) * 0.3 - 2.5)

    @property
    def Lambda(self):
        return torch.exp(self.log_lambda)

    def regularization_loss(self, r_public):
        r_public = torch.as_tensor(r_public, dtype=self.r_j.dtype,
                                   device=self.r_j.device)
        diff = self.r_j - r_public
        return (diff * self.Lambda * diff).sum()


# ---------------------------------------------------------------------------
# BayesianInverse  z*  (real action -> intent-correction vector)
# ---------------------------------------------------------------------------

class BayesianInverse(nn.Module):
    """z* = tanh(z_prior + strength * direction).

        direction = normalize(W_dir(s))                 <- learned
        strength  = 0.05 + 0.5 * sigmoid(W_mag(s))      <- learned

    s is the action signal (M dims). This is the v5 "Bayesian inverse of the
    real action": observing what was actually played corrects the prior intent.
    No keyword classifier, no discrete response_type -- the whole map is trained,
    so the KL/dissonance loss never inherits a labelling error.
    """

    def __init__(self, d=config.D, m=config.M):
        super().__init__()
        self.d = d
        self.m = m
        self.W_dir = nn.Linear(m, d, bias=False)
        self.W_mag = nn.Linear(m, 1, bias=True)

    def forward(self, s, z_prior):
        direction = F.normalize(self.W_dir(s), dim=-1, eps=1e-8)
        strength = 0.05 + 0.5 * torch.sigmoid(self.W_mag(s)).squeeze(-1)
        return torch.tanh(z_prior + strength * direction)


# ---------------------------------------------------------------------------
# ToleranceHead  (responder's resistance threshold from its intent vector)
# ---------------------------------------------------------------------------

class ToleranceHead(nn.Module):
    """tolerance(z) = tol_min + sigmoid(W z + b) (tol_max - tol_min).

    Reads the *whole* z (no slicing). Used by the future tree to modulate how
    likely the responder is to reject a given bid.
    """

    def __init__(self, d=config.D, tol_min=config.TOL_MIN, tol_max=config.TOL_MAX):
        super().__init__()
        self.W = nn.Linear(d, 1, bias=True)
        self.register_buffer("tol_min", torch.tensor(float(tol_min)))
        self.register_buffer("tol_max", torch.tensor(float(tol_max)))

    def forward(self, z_j):
        t = torch.sigmoid(self.W(z_j)).squeeze(-1)
        return self.tol_min + t * (self.tol_max - self.tol_min)


def dissonance_loss(z, z_star):
    """L = KL(softmax(z) || softmax(z*)) -- cognitive dissonance over a row."""
    log_p = F.log_softmax(z, dim=-1)
    q = F.softmax(z_star, dim=-1)
    return F.kl_div(log_p, q, reduction="sum")
