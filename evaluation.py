"""Architecture verification probes.

The core architectural claim is that understanding is useful only when it
changes action selection or expected future position. This module compares
normal deliberation against ablated deliberation to make that claim measurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from runtime import ActionEvent, WorldResponse


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


@dataclass(frozen=True)
class ArchitectureGateReport:
    """Deployment-style health gate for runtime-capable applications."""

    legal_action_rate: float
    episode_completion_rate: float
    response_diversity: int
    ablation_delta: float | None
    errors: tuple[str, ...] = ()

    @property
    def passed(self):
        return not self.errors


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


def check_action_response_separation(action_event, world_response):
    if not isinstance(action_event, ActionEvent):
        raise TypeError("Expected ActionEvent.")
    if not isinstance(world_response, WorldResponse):
        raise TypeError("Expected WorldResponse.")
    if action_event.label == world_response.label:
        return False
    return True


def runtime_gate_report(action_events, world_responses, terminal_count,
                        episode_count, ablation_delta=None):
    errors = []
    n_actions = max(len(action_events), 1)
    legal = [
        bool(getattr(event, "metadata", {}).get("legal", True))
        for event in action_events
    ]
    legal_action_rate = sum(legal) / n_actions
    if legal_action_rate < 1.0:
        errors.append("illegal_action")
    response_labels = [response.label for response in world_responses]
    response_diversity = len(set(response_labels))
    if world_responses and response_diversity < 1:
        errors.append("no_world_response")
    completion = terminal_count / max(episode_count, 1)
    if episode_count and completion <= 0.0:
        errors.append("no_completed_episode")
    return ArchitectureGateReport(
        legal_action_rate=legal_action_rate,
        episode_completion_rate=completion,
        response_diversity=response_diversity,
        ablation_delta=ablation_delta,
        errors=tuple(errors),
    )
