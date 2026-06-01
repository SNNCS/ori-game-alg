"""Decision layer for closing the perception -> action loop.

The original G/I/T stack can interpret an externally supplied action and
simulate futures. This module adds the missing decision surface:

    candidate interventions -> predicted futures -> utility -> chosen action

Outcome is intentionally not represented here. The decision layer scores
counterfactual futures before acting; later learning code should compare those
predictions against observed outcomes and realized utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CandidateIntervention:
    """A possible thing the focal agent can do.

    `action` is the adapter-decoded executable move. `latent_action` is the
    internal generated action vector before grounding. `signal` is the learned
    communicative part of the intervention.
    """

    action: Any
    signal: Any = None
    latent_action: torch.Tensor | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PredictedFuture:
    """Counterfactual future conditioned on one candidate intervention."""

    candidate: CandidateIntervention
    outgoing_signal: Any
    tree: Any
    metrics: Mapping[str, torch.Tensor]
    signal_vec: torch.Tensor
    Z: torch.Tensor
    z_C: torch.Tensor


@dataclass(frozen=True)
class DecisionResult:
    """Decision output plus differentiable training quantities."""

    selected: CandidateIntervention
    selected_future: PredictedFuture
    futures: Sequence[PredictedFuture]
    scores: torch.Tensor
    action_probs: torch.Tensor
    expected_utility: torch.Tensor
    selected_index: int


class FuturePositionEvaluator(nn.Module):
    """Learnable utility interface over planner metrics.

    This keeps raw planner metrics separate from utility. The initial weights
    are uniform; training can later discover which future-position dimensions
    matter for the agent's own objective.
    """

    DEFAULT_METRICS = ("path_quality", "optionality", "risk_floor")

    def __init__(self, metric_names=DEFAULT_METRICS):
        super().__init__()
        self.metric_names = tuple(metric_names)
        self.raw_weights = nn.Parameter(torch.zeros(len(self.metric_names)))

    @property
    def weights(self):
        return F.softmax(self.raw_weights, dim=0)

    def forward(self, metrics):
        values = torch.stack([metrics[name] for name in self.metric_names])
        return (self.weights.to(values.device, values.dtype) * values).sum()


class DecisionEngine(nn.Module):
    """Selects an intervention by scoring predicted futures."""

    def __init__(self, temperature=0.25):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, futures, utility_model):
        if not futures:
            raise ValueError("DecisionEngine requires at least one future.")

        scores = torch.stack([utility_model(f.metrics) for f in futures])
        temp = max(self.temperature, 1e-6)
        action_probs = F.softmax(scores / temp, dim=0)
        expected_utility = (action_probs * scores).sum()
        selected_index = int(torch.argmax(scores.detach()).item())
        return DecisionResult(
            selected=futures[selected_index].candidate,
            selected_future=futures[selected_index],
            futures=tuple(futures),
            scores=scores,
            action_probs=action_probs,
            expected_utility=expected_utility,
            selected_index=selected_index,
        )
