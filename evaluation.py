"""Architecture verification probes.

The core architectural claim is that understanding is useful only when it
changes action selection or expected future position. This module compares
normal deliberation against ablated deliberation to make that claim measurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AblationSpec:
    """Switches off parts of the internal model during deliberation."""

    name: str
    zero_intent: bool = False
    zero_signal: bool = False
    zero_actor_situation: bool = False


@dataclass(frozen=True)
class UsefulnessReport:
    """Comparison between full and ablated deliberation."""

    ablation: AblationSpec
    full_action: float
    ablated_action: float
    full_expected_utility: torch.Tensor
    ablated_expected_utility: torch.Tensor
    utility_delta: torch.Tensor
    score_delta_norm: torch.Tensor
    action_prob_delta_norm: torch.Tensor
    action_changed: bool


NO_UNDERSTANDING = AblationSpec(
    name="no_understanding",
    zero_intent=True,
    zero_signal=True,
    zero_actor_situation=True,
)

NO_SIGNAL = AblationSpec(name="no_signal", zero_signal=True)


def compare_decisions(full_decision, ablated_decision, ablation):
    full_action = float(full_decision.selected.action)
    ablated_action = float(ablated_decision.selected.action)
    utility_delta = (
        full_decision.expected_utility -
        ablated_decision.expected_utility
    )
    return UsefulnessReport(
        ablation=ablation,
        full_action=full_action,
        ablated_action=ablated_action,
        full_expected_utility=full_decision.expected_utility,
        ablated_expected_utility=ablated_decision.expected_utility,
        utility_delta=utility_delta,
        score_delta_norm=torch.linalg.norm(
            full_decision.scores - ablated_decision.scores),
        action_prob_delta_norm=torch.linalg.norm(
            full_decision.action_probs - ablated_decision.action_probs),
        action_changed=full_action != ablated_action,
    )
