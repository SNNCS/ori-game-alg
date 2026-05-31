"""UltimatumRule -- the payoff / legality structure of the ultimatum game.

This is the pure game mechanic, kept deliberately small and self-contained
(no SentenceTransformer rule-text encoder, no e-commerce outcomes).

Payoff (v5 spec):
    accept : proposer (role A) keeps `action`, responder (role B) gets 1-action
    reject : both get the outside option (the classic "spite" outcome)
    counter: no immediate payoff; the pie is discounted and play continues

`r_public` is the public/normative rule interpretation that RuleInterpretation
regularises toward. phi/psi are exposed as plain tensors so the legality band
stays differentiable-friendly without depending on any external model.
"""

import torch

import config


class UltimatumRule:
    def __init__(self, p=config.P,
                 outside=config.OUTSIDE_OPTION,
                 hard_center=0.5, hard_width=0.4,
                 soft_thresh=0.25, dtype=torch.float32):
        self.p = p
        self.outside = float(outside)
        self.hard_center = float(hard_center)
        self.hard_width = float(hard_width)
        self.soft_thresh = float(soft_thresh)
        self.dtype = dtype
        # Public rule stance: a fair-split prior. RuleInterpretation.r_j drifts
        # away from this under pressure; the regulariser pulls it back.
        self.r_public = torch.zeros(p, dtype=dtype)
        self.r_public[0] = 0.5

    # ----- payoffs ---------------------------------------------------------

    def compute_payoff(self, action, response, role_id, pie=1.0):
        """action = proposer's kept share in [0,1]; pie = current stake."""
        action = float(action)
        pie = float(pie)
        if response == "accept":
            keep = pie * action
            return keep if role_id == config.ACTOR_A else pie * (1.0 - action)
        if response == "reject":
            return self.outside
        return 0.0   # counter: deferred to the next round

    # ----- legality (offer boundary) --------------------------------------

    def is_legal(self, action, r_j=None):
        if action < 0.0 or action > 1.0:
            return False
        if abs(action - self.hard_center) <= self.hard_width:
            return True
        if r_j is None:
            return False
        # Outside the hard band, an offer is only legal if the proposer's rule
        # stance has drifted far enough from the public norm to justify it.
        deviation = float(torch.linalg.norm(
            r_j.detach() - self.r_public.to(r_j.device)).item())
        return deviation > self.soft_thresh

    def is_soft_action(self, action):
        return 0.0 <= action <= 1.0 and abs(action - self.hard_center) > self.hard_width
