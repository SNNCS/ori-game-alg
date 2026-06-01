"""Post-action experience interfaces.

This module keeps three ideas separate:

    Outcome          raw facts about what happened
    RealizedUtility  a learned evaluation of those facts
    LearningSignal   prediction/value/policy losses for credit assignment

The split is important: an outcome is not a reward. Utility is a model over
outcome features, and prediction error is its own learning signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


@dataclass(frozen=True)
class Outcome:
    """Raw observed result after an intervention."""

    action: float
    response: str
    pie_after: float
    paths_open: float
    terminal: bool
    raw_state: Mapping[str, Any]
    entity_payoffs: Mapping[int, float] | None = None
    payoff_A: float | None = None
    payoff_B: float | None = None

    def payoff_for(self, role, adapter=None):
        if self.entity_payoffs is not None and role in self.entity_payoffs:
            return self.entity_payoffs[role]
        if (
            adapter is not None
            and role == adapter.counterpart_actor
            and self.payoff_B is not None
        ):
            return self.payoff_B
        return 0.0 if self.payoff_A is None else self.payoff_A

    def payoff_against(self, role, adapter=None):
        if self.entity_payoffs is not None:
            other = None
            if adapter is not None:
                if role == adapter.focal_actor:
                    other = adapter.counterpart_actor
                elif role == adapter.counterpart_actor:
                    other = adapter.focal_actor
            if other is None:
                other = next(
                    (entity for entity in self.entity_payoffs if entity != role),
                    None)
            if other in self.entity_payoffs:
                return self.entity_payoffs[other]
        if (
            adapter is not None
            and role == adapter.counterpart_actor
            and self.payoff_A is not None
        ):
            return self.payoff_A
        return 0.0 if self.payoff_B is None else self.payoff_B


@dataclass(frozen=True)
class RealizedUtility:
    """Utility model output for an observed outcome."""

    value: torch.Tensor
    features: torch.Tensor
    weights: torch.Tensor
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class LearningSignal:
    """Separated losses for post-action credit assignment."""

    prediction_loss: torch.Tensor
    value_loss: torch.Tensor
    policy_loss: torch.Tensor
    total_loss: torch.Tensor


@dataclass(frozen=True)
class ExperienceStep:
    """A complete acted-and-observed step."""

    decision: object
    outcome: Outcome
    realized_utility: RealizedUtility
    learning_signal: LearningSignal


class OutcomeFeatureEncoder:
    """Adapter-backed conversion from raw outcomes into utility features."""

    def __init__(self, adapter=None):
        if adapter is None:
            from game_rule import UltimatumRule
            from game_adapter import UltimatumGameAdapter
            adapter = UltimatumGameAdapter(UltimatumRule())
        self.adapter = adapter
        self.FEATURE_NAMES = tuple(adapter.outcome_feature_names)

    def __call__(self, outcome, role=None,
                 device=None, dtype=torch.float32):
        if role is None:
            role = self.adapter.focal_actor
        return self.adapter.outcome_features(
            outcome, role=role, device=device, dtype=dtype)


class OutcomeUtilityEvaluator(nn.Module):
    """Learnable utility interface over observed outcome features."""

    def __init__(self, encoder=None, adapter=None):
        super().__init__()
        self.encoder = (
            encoder if encoder is not None else OutcomeFeatureEncoder(adapter))
        self.feature_names = tuple(self.encoder.FEATURE_NAMES)
        self.raw_weights = nn.Parameter(torch.zeros(len(self.feature_names)))

    @property
    def weights(self):
        return F.softmax(self.raw_weights, dim=0)

    def forward(self, outcome, role=None,
                device=None, dtype=torch.float32):
        if role is None:
            role = self.encoder.adapter.focal_actor
        features = self.encoder(outcome, role=role, device=device, dtype=dtype)
        weights = self.weights.to(device=features.device, dtype=features.dtype)
        value = (weights * features).sum()
        return RealizedUtility(
            value=value,
            features=features,
            weights=weights,
            feature_names=self.feature_names,
        )


def resolve_ultimatum_outcome(rule, action, response, pie=1.0, adapter=None):
    """Resolve raw game facts for one observed response."""
    if adapter is None:
        from game_adapter import UltimatumGameAdapter
        adapter = UltimatumGameAdapter(rule)
    return adapter.resolve_outcome(action, response, pie=pie)


def response_prediction_loss(predicted_future, outcome):
    """Cross-entropy over the selected future's root response branches."""
    children = predicted_future.tree.children
    responses = [child.response for child in children]
    if outcome.response not in responses:
        raise ValueError(f"Unknown response in outcome: {outcome.response}")
    probs = torch.stack([child.prob for child in children])
    probs = probs / (probs.sum() + 1e-8)
    target = torch.tensor(
        [responses.index(outcome.response)],
        device=probs.device,
        dtype=torch.long,
    )
    return F.nll_loss(torch.log(probs + 1e-8).reshape(1, -1), target)


def build_learning_signal(decision, outcome, realized_utility):
    """Build separated learning losses for one acted-and-observed step."""
    pred_loss = response_prediction_loss(decision.selected_future, outcome)
    predicted_value = decision.scores[decision.selected_index]
    value_loss = F.mse_loss(predicted_value, realized_utility.value.detach())
    policy_loss = -decision.expected_utility
    total = pred_loss + value_loss + policy_loss
    return LearningSignal(
        prediction_loss=pred_loss,
        value_loss=value_loss,
        policy_loss=policy_loss,
        total_loss=total,
    )
