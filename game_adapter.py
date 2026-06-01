"""Domain adapters for concrete environments.

The core agent should not own domain-specific roles, action enumerations,
response labels, or outcome feature names. This module keeps the current
Ultimatum Game details behind an adapter interface, so later environments can
provide different entities, interventions, observations, and utility features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

import config


@dataclass(frozen=True)
class EntitySet:
    """Concrete entities participating in an environment."""

    focal: int
    counterpart: int
    observer: int | None
    n_entities: int


@dataclass(frozen=True)
class ActionAffordance:
    """Executable control boundary exposed by an environment."""

    low: float
    high: float
    n_candidates: int
    control_name: str = "action"


class UltimatumGameAdapter:
    """Adapter for the current Ultimatum Game environment.

    This is where toy-game-specific discreteness lives: three entities,
    candidate bid values, response labels, and outcome feature names.
    """

    def __init__(self, rule, bids=config.BIDS, responses=config.RESPONSES):
        self.rule = rule
        self.entities = EntitySet(
            focal=config.ACTOR_A,
            counterpart=config.ACTOR_B,
            observer=config.ACTOR_C,
            n_entities=config.N_AGENTS,
        )
        self._bids = tuple(float(b) for b in bids)
        self._responses = tuple(responses)
        self.outcome_feature_names = (
            "self_payoff",
            "other_payoff",
            "paths_open",
            "terminal",
            "accepted",
            "rejected",
            "countered",
            "fairness_deviation",
        )

    @property
    def focal_actor(self):
        return self.entities.focal

    @property
    def counterpart_actor(self):
        return self.entities.counterpart

    @property
    def observer_actor(self):
        return self.entities.observer

    def candidate_interventions(self, state=None) -> Sequence[float]:
        """Return candidate physical interventions for this domain."""
        return self._bids

    def action_affordance(self, state=None) -> ActionAffordance:
        """Return the current controllable action boundary.

        The generator creates latent actions; this adapter decodes the first
        latent coordinate into a continuous kept-share bid within this range.
        """
        return ActionAffordance(
            low=min(self._bids),
            high=max(self._bids),
            n_candidates=len(self._bids),
            control_name="kept_share",
        )

    def decode_action(self, latent_action, affordance=None):
        affordance = (
            affordance if affordance is not None
            else self.action_affordance())
        z = torch.as_tensor(latent_action)
        low = torch.as_tensor(affordance.low, device=z.device, dtype=z.dtype)
        high = torch.as_tensor(affordance.high, device=z.device, dtype=z.dtype)
        control = torch.sigmoid(z.reshape(-1)[0])
        return low + (high - low) * control

    def validate_intervention(self, action) -> bool:
        action_value = float(torch.as_tensor(action).detach().cpu())
        return self.rule.is_legal(action_value)

    def response_labels(self) -> Sequence[str]:
        """Return observable response labels for this domain."""
        return self._responses

    def initial_resource_map(self):
        return {i: 0.0 for i in range(self.entities.n_entities)}

    def initial_knowledge(self, dim=config.K_DIM, dtype=torch.float32):
        """Public knowledge for this concrete domain."""
        K = torch.zeros(dim, dtype=dtype)
        K[0] = 1.0     # total pie known to be 1
        K[1] = 1.0     # both players are rational observers
        K[2] = 1.0     # the responder may reject
        K[3] = 0.6     # observability of the offer
        if dim >= 8:
            K[4:8] = K[0:4] * 0.7
        return K

    def is_continue_response(self, response):
        return response == "counter"

    def intent_shift_for_response(self, response):
        if response == "reject":
            return 0.05
        if response == "accept":
            return -0.05
        return 0.0

    def initial_tree_state(self, sigma_root=None):
        return {"payoff_A": 0.0, "payoff_B": 0.0,
                "pie": 1.0, "paths_open": 1.0, "sigma": sigma_root}

    def transition(self, state, action, response):
        new = dict(state)
        pie = state["pie"]
        action_value = float(torch.as_tensor(action).detach().cpu())
        if response == "accept":
            new["payoff_A"] = state["payoff_A"] + self.rule.compute_payoff(
                action_value, "accept", self.focal_actor, pie)
            new["payoff_B"] = state["payoff_B"] + self.rule.compute_payoff(
                action_value, "accept", self.counterpart_actor, pie)
            new["paths_open"] = 0.0
        elif response == "reject":
            new["payoff_A"] = state["payoff_A"] + self.rule.outside
            new["payoff_B"] = state["payoff_B"] + self.rule.outside
            new["paths_open"] = 0.0
        else:
            new["pie"] = pie * config.COUNTER_DISCOUNT
            new["paths_open"] = max(
                0.0, state["paths_open"] - config.PATHS_OPEN_DECAY)
        return new

    def outcome_quality(self, state, role):
        payoff_key = "payoff_A" if role == self.focal_actor else "payoff_B"
        return state[payoff_key] + 0.2 * state["paths_open"]

    def resolve_outcome(self, action, response, pie=1.0):
        action = float(torch.as_tensor(action).detach().cpu())
        pie = float(pie)
        payoff_A = float(self.rule.compute_payoff(
            action, response, self.focal_actor, pie))
        payoff_B = float(self.rule.compute_payoff(
            action, response, self.counterpart_actor, pie))
        if response == "counter":
            pie_after = pie * config.COUNTER_DISCOUNT
            paths_open = max(0.0, 1.0 - config.PATHS_OPEN_DECAY)
            terminal = False
        else:
            pie_after = pie
            paths_open = 0.0
            terminal = True
        from experience import Outcome
        return Outcome(
            action=action,
            response=response,
            payoff_A=payoff_A,
            payoff_B=payoff_B,
            pie_after=pie_after,
            paths_open=paths_open,
            terminal=terminal,
            raw_state={
                "pie": pie_after,
                "paths_open": paths_open,
                "payoff_A": payoff_A,
                "payoff_B": payoff_B,
            },
            entity_payoffs={
                self.focal_actor: payoff_A,
                self.counterpart_actor: payoff_B,
            },
        )

    def outcome_features(self, outcome, role,
                         device=None, dtype=torch.float32):
        response = outcome.response
        action = float(outcome.action)
        values = [
            float(outcome.payoff_for(role, adapter=self)),
            float(outcome.payoff_against(role, adapter=self)),
            float(outcome.paths_open),
            1.0 if outcome.terminal else 0.0,
            1.0 if response == "accept" else 0.0,
            1.0 if response == "reject" else 0.0,
            1.0 if response == "counter" else 0.0,
            2.0 * abs(action - 0.5),
        ]
        return torch.tensor(values, device=device, dtype=dtype)
