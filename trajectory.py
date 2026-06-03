"""Trajectory-level learning contracts.

Complex applications often resolve utility after several events. This module
keeps trajectory targets distinct from single-step outcomes while preserving the
existing ExperienceStep path as the length-one case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from experience import (
    LearningSignal, build_learning_signal, response_prediction_loss,
)
from runtime import ActionEvent, TerminalOutcome, TransitionResult, WorldResponse


@dataclass(frozen=True)
class TrajectoryStep:
    """One acted-and-observed runtime step."""

    decision: object
    action_event: ActionEvent
    world_response: WorldResponse
    transition: TransitionResult
    outcome: object | None = None
    realized_utility: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """A sequence of runtime steps ending optionally in a terminal outcome."""

    steps: Sequence[TrajectoryStep]
    terminal_outcome: TerminalOutcome | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_terminal(self):
        return self.terminal_outcome is not None


@dataclass(frozen=True)
class ReturnTarget:
    """Role-relative value target for a trajectory."""

    value: torch.Tensor
    role: object
    terminal: bool
    normalization: float = 1.0
    available: bool = True


def _resolve_role(role, adapter=None):
    if adapter is not None:
        return int(adapter.resolve_role(role))
    try:
        return int(role)
    except (TypeError, ValueError):
        return role


class ReturnBuilder:
    """Build role-relative terminal or continuation returns."""

    def __call__(self, trajectory: Trajectory, role, device=None,
                 dtype=torch.float32, adapter=None):
        resolved_role = _resolve_role(role, adapter=adapter)
        if trajectory.terminal_outcome is None:
            continuation = trajectory.metadata.get("continuation_value")
            if isinstance(continuation, Mapping):
                continuation = (
                    continuation.get(role)
                    if role in continuation
                    else continuation.get(resolved_role))
            if continuation is not None:
                value = torch.as_tensor(
                    continuation, device=device, dtype=dtype).reshape(())
                return ReturnTarget(
                    value=value, role=resolved_role, terminal=False,
                    available=True)
            value = torch.tensor(0.0, device=device, dtype=dtype)
            return ReturnTarget(
                value=value, role=resolved_role, terminal=False,
                available=False)
        payoff = trajectory.terminal_outcome.payoff_for(
            role, adapter=adapter)
        value = torch.as_tensor(payoff, device=device, dtype=dtype).reshape(())
        return ReturnTarget(
            value=value, role=resolved_role, terminal=True, available=True)


class OutcomeTargetBuilder:
    """Build utility targets without treating raw events as rewards."""

    def __init__(self, return_builder=None, normalization=1.0):
        self.return_builder = return_builder if return_builder is not None else ReturnBuilder()
        self.normalization = float(normalization)

    def __call__(self, trajectory: Trajectory, role, device=None,
                 dtype=torch.float32, adapter=None):
        target = self.return_builder(
            trajectory, role, device=device, dtype=dtype, adapter=adapter)
        value = target.value / max(self.normalization, 1e-8)
        return ReturnTarget(
            value=value,
            role=target.role,
            terminal=target.terminal,
            normalization=self.normalization,
            available=target.available,
        )


class LearningCoordinator:
    """Create separated prediction/value/policy losses for trajectories."""

    def __init__(self, target_builder=None):
        self.target_builder = (
            target_builder if target_builder is not None
            else OutcomeTargetBuilder())

    def build_single_step_signal(self, decision, outcome, realized_utility):
        return build_learning_signal(decision, outcome, realized_utility)

    def build_trajectory_signal(self, trajectory: Trajectory, role,
                                utility_evaluator=None, adapter=None):
        if not trajectory.steps:
            raise ValueError("Trajectory requires at least one step.")
        for step in trajectory.steps:
            if step.outcome is None or step.realized_utility is None:
                raise ValueError(
                    "TrajectoryStep requires outcome and realized_utility for now.")

        first_step = trajectory.steps[0]
        predicted_value = first_step.decision.scores[
            first_step.decision.selected_index]
        target = self.target_builder(
            trajectory,
            role,
            device=predicted_value.device,
            dtype=predicted_value.dtype,
            adapter=adapter,
        )
        target_value = target.value.detach()

        prediction_losses = []
        value_losses = []
        policy_losses = []
        fallback_policy_losses = []
        for step in trajectory.steps:
            selected_value = step.decision.scores[step.decision.selected_index]
            prediction_losses.append(response_prediction_loss(
                step.decision.selected_future, step.world_response))
            fallback_policy_losses.append(-step.decision.expected_utility)
            if target.available:
                value_losses.append(F.mse_loss(selected_value, target_value))
                log_prob = step.action_event.log_prob
                if log_prob is not None:
                    advantage = target_value - selected_value.detach()
                    policy_losses.append(
                        -(log_prob.to(selected_value.dtype) * advantage))
        prediction_loss = torch.stack(prediction_losses).mean()
        if target.available:
            value_loss = torch.stack(value_losses).mean()
            if policy_losses:
                policy_loss = torch.stack(policy_losses).mean()
            else:
                policy_loss = torch.stack(fallback_policy_losses).mean()
        else:
            value_loss = predicted_value.new_zeros(())
            policy_loss = torch.stack(fallback_policy_losses).mean()
        total = prediction_loss + value_loss + policy_loss
        return LearningSignal(
            prediction_loss=prediction_loss,
            value_loss=value_loss,
            policy_loss=policy_loss,
            total_loss=total,
        )
