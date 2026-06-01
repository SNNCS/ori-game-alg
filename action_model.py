"""Learnable candidate-action generation.

The adapter describes what can be controlled in a concrete domain. This module
decides which latent interventions to try. A latent action only becomes an
action after the adapter decodes it into an executable intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

import config
from decision import CandidateIntervention
from interpretation import build_context


@dataclass(frozen=True)
class GeneratedInterventions:
    """Generated latent actions plus their decoded interventions."""

    candidates: Sequence[CandidateIntervention]
    latent_actions: torch.Tensor
    context_vector: torch.Tensor


class CandidateInterventionGenerator(nn.Module):
    """Generate latent actions and ground them through a domain adapter.

    Input:
      actor situation, counterpart situation, relation edge, public context.

    Output:
      several latent action vectors, each decoded by the adapter into a
      concrete intervention that can be simulated by the future tree.
    """

    def __init__(self, latent_dim=config.ACTION_LATENT_DIM,
                 n_candidates=config.N_GENERATED_ACTIONS,
                 input_dim=config.ACTION_CONTEXT_DIM,
                 hidden_dim=config.ACTION_HIDDEN_DIM):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_candidates = int(n_candidates)
        self.input_dim = int(input_dim)
        self.context_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        slots = torch.zeros(n_candidates, latent_dim)
        slots[:, 0] = torch.linspace(-1.5, 1.5, n_candidates)
        if latent_dim > 1:
            slots[:, 1:] = 0.05 * torch.randn(n_candidates, latent_dim - 1)
        self.slot_latents = nn.Parameter(slots)

    def _context_tensor(self, context, device, dtype):
        if context is None:
            context = build_context()
        return torch.as_tensor(context, device=device, dtype=dtype)

    def forward(self, actor_sigma, counterpart_sigma, relation_edge, context,
                adapter, n_candidates=None):
        if n_candidates is None and hasattr(adapter, "action_affordance"):
            n_candidates = adapter.action_affordance().n_candidates
        n = int(n_candidates or self.n_candidates)
        if n < 1 or n > self.n_candidates:
            raise ValueError(
                f"n_candidates must be in [1, {self.n_candidates}], got {n}")

        ctx = self._context_tensor(
            context, device=actor_sigma.device, dtype=actor_sigma.dtype)
        x = torch.cat([actor_sigma, counterpart_sigma, relation_edge, ctx],
                      dim=-1)
        if x.shape[-1] != self.input_dim:
            raise ValueError((x.shape, self.input_dim))

        base = torch.tanh(self.context_net(x))
        latent_actions = torch.tanh(self.slot_latents[:n] + base.unsqueeze(0))
        candidates = tuple(
            CandidateIntervention(
                action=adapter.decode_action(latent),
                latent_action=latent,
                metadata={"source": "generated"},
            )
            for latent in latent_actions
        )
        return GeneratedInterventions(
            candidates=candidates,
            latent_actions=latent_actions,
            context_vector=x,
        )
