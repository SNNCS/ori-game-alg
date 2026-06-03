"""Typed runtime contracts for complex game applications.

The original GameSpec path describes compact games well. These contracts add a
runtime layer for richer environments where current state, observations, legal
actions, world responses, and terminal outcomes need explicit types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class RuntimeSnapshot:
    """A role-relative environment state at one decision point."""

    state: Mapping[str, Any]
    current_actor: int | None = None
    step_index: int = 0
    terminal: bool = False
    public: Mapping[str, Any] = field(default_factory=dict)
    private: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    legal_action_mask: Mapping[int, Sequence[bool]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationSpec:
    """Declared observation layout for a viewer."""

    feature_names: tuple[str, ...]
    private_feature_names: tuple[str, ...] = ()
    dim: int | None = None
    schema_version: str = "1"


@dataclass(frozen=True)
class Observation:
    """What a viewer can observe from a runtime snapshot."""

    viewer: int
    vector: torch.Tensor
    mask: torch.Tensor
    spec: ObservationSpec
    public_state: Mapping[str, Any]
    private_state: Mapping[str, Any] = field(default_factory=dict)
    snapshot_step: int = 0


@dataclass(frozen=True)
class ActionEvent:
    """What the acting agent actually did to the runtime."""

    actor: int
    action: Any
    label: str
    controls: Mapping[str, Any] = field(default_factory=dict)
    log_prob: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __float__(self):
        return float(self.action)


@dataclass(frozen=True)
class WorldResponse:
    """What the world or other agents did after an ActionEvent."""

    label: str
    source: int | None = None
    target: int | None = None
    features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TerminalOutcome:
    """A resolved terminal result with role-relative payoffs."""

    snapshot: RuntimeSnapshot
    payoffs: Mapping[int, Any]
    terminal: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def payoff_for(self, role, adapter=None):
        payoffs = dict(self.payoffs or {})
        if adapter is not None:
            role = adapter.resolve_role(role)
        if role in payoffs:
            return payoffs[role]
        try:
            role_id = int(role)
        except (TypeError, ValueError):
            return payoffs.get(str(role), 0.0)
        return payoffs.get(role_id, 0.0)


@dataclass(frozen=True)
class TransitionResult:
    """Result of applying an ActionEvent and observing a WorldResponse."""

    before: RuntimeSnapshot
    action_event: ActionEvent
    world_response: WorldResponse
    after: RuntimeSnapshot
    terminal_outcome: TerminalOutcome | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeSchema:
    """Schema metadata used for checkpoint and application compatibility."""

    spec_name: str
    spec_version: str = "1"
    observation_features: tuple[str, ...] = ()
    action_controls: tuple[str, ...] = ()
    world_response_labels: tuple[str, ...] = ()
    outcome_features: tuple[str, ...] = ()
    n_entities: int | None = None


@dataclass(frozen=True)
class CheckpointMetadata:
    """Minimal checkpoint-side schema declaration."""

    runtime_schema: RuntimeSchema
    head_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    training_status: str = "unknown"


@dataclass(frozen=True)
class SchemaCompatibilityReport:
    """Result of comparing a checkpoint schema to the active runtime schema."""

    compatible: bool
    mismatches: tuple[str, ...] = ()


def check_schema_compatibility(active: RuntimeSchema,
                               checkpoint: CheckpointMetadata):
    """Return explicit mismatches instead of silently accepting stale heads."""
    saved = checkpoint.runtime_schema
    mismatches: list[str] = []
    fields = (
        "spec_name",
        "spec_version",
        "observation_features",
        "action_controls",
        "world_response_labels",
        "outcome_features",
        "n_entities",
    )
    for field_name in fields:
        if getattr(active, field_name) != getattr(saved, field_name):
            mismatches.append(field_name)
    return SchemaCompatibilityReport(
        compatible=not mismatches,
        mismatches=tuple(mismatches),
    )
