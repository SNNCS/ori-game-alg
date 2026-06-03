"""Explicit belief-state container for the closed-loop agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from runtime import Observation


@dataclass(frozen=True)
class BeliefState:
    """The agent's internal model after reading an observation.

    `Z` is optional because intent is action-conditioned: an observation can
    update situation, resources, history, and public/private state without
    implying that any particular action has been interpreted.
    """

    observation: Observation
    actor: int
    observation_embedding: torch.Tensor
    sigma: Mapping[int, torch.Tensor]
    rules: Mapping[int, torch.Tensor]
    action_signal: torch.Tensor | None = None
    Z: torch.Tensor | None = None
    z_C: torch.Tensor | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def intent_for(self, role):
        if self.Z is None:
            raise ValueError("BeliefState has no action-conditioned intent matrix.")
        return self.Z[role]
