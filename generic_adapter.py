"""Generic adapter that interprets declarative GameSpec objects."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

import config
from experience import Outcome
from game_spec import (
    ActionAffordance, EntitySet, GameSpec, GroundedAction,
)


class GenericRule:
    """Minimal rule facade expected by CognitiveAgent."""

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.outside = 0.0
        self.r_public = torch.zeros(config.P, dtype=torch.float32)
        self.r_public[0] = float(spec.r_public_first)

    def is_legal(self, action, r_j=None):
        return True


class GenericGameAdapter:
    """GameSpec interpreter exposing the adapter contract used by the agent."""

    action_signal_dim = config.M
    branch_action_feature_dim = 3

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.rule = GenericRule(spec)
        self.entity_names = tuple(entity.name for entity in spec.entities)
        self.entity_index = {
            name: idx for idx, name in enumerate(self.entity_names)
        }
        self.entities = EntitySet(
            focal=self.resolve_role(spec.roles.focal),
            counterpart=self.resolve_role(spec.roles.counterpart),
            observer=(
                None if spec.roles.observer is None
                else self.resolve_role(spec.roles.observer)
            ),
            n_entities=len(self.entity_names),
        )
        self.outcome_feature_names = tuple(
            feature.name for feature in spec.outcome_features)
        self._responses = {response.label: response
                           for response in spec.responses}
        self._transitions = {transition.response: transition
                             for transition in spec.transitions}

    @property
    def focal_actor(self):
        return self.entities.focal

    @property
    def counterpart_actor(self):
        return self.entities.counterpart

    @property
    def observer_actor(self):
        return self.entities.observer

    def resolve_role(self, role):
        if isinstance(role, int):
            return role
        if role == "focal":
            return self.entity_index[self.spec.roles.focal]
        if role == "counterpart":
            return self.entity_index[self.spec.roles.counterpart]
        if role == "observer":
            if self.spec.roles.observer is None:
                raise ValueError("This spec has no observer role.")
            return self.entity_index[self.spec.roles.observer]
        return self.entity_index[role]

    def response_labels(self):
        return tuple(response.label for response in self.spec.responses)

    def action_affordance(self, state=None):
        primary = self.spec.action_controls[0]
        return ActionAffordance(
            low=float(primary.low),
            high=float(primary.high),
            n_candidates=int(self.spec.n_candidates),
            control_name=primary.name,
        )

    def _decode_control(self, spec, latent, idx):
        if spec.kind == "continuous":
            raw = latent[idx]
            low = torch.as_tensor(spec.low, device=latent.device, dtype=latent.dtype)
            high = torch.as_tensor(spec.high, device=latent.device, dtype=latent.dtype)
            return low + (high - low) * torch.sigmoid(raw), idx + 1
        if spec.kind == "binary":
            return torch.sigmoid(latent[idx]), idx + 1
        if spec.kind == "categorical":
            n = len(spec.categories)
            if n < 1:
                raise ValueError(f"Categorical control {spec.name} has no labels.")
            return torch.softmax(latent[idx:idx + n], dim=0), idx + n
        raise ValueError(f"Unknown control kind: {spec.kind}")

    def decode_action(self, latent_action, affordance=None):
        latent = torch.as_tensor(latent_action)
        idx = 0
        controls: dict[str, Any] = {}
        display_bits = []
        for control_spec in self.spec.action_controls:
            value, idx = self._decode_control(control_spec, latent, idx)
            controls[control_spec.name] = value
            if control_spec.kind == "categorical":
                label_idx = int(torch.argmax(value.detach()).item())
                label = control_spec.categories[label_idx]
                display_bits.append(f"{control_spec.name}={label}")
            else:
                display_bits.append(
                    f"{control_spec.name}={float(value.detach()):.3f}")

        primary_name = self.spec.action_controls[0].name
        return GroundedAction(
            controls=controls,
            primary_value=controls[primary_name],
            display=", ".join(display_bits),
            metadata={"spec": self.spec.name},
        )

    def _coerce_action(self, action):
        if isinstance(action, GroundedAction):
            return action
        primary = self.spec.action_controls[0]
        value = torch.as_tensor(action, dtype=torch.float32)
        return GroundedAction(
            controls={primary.name: value},
            primary_value=value,
            display=f"{primary.name}={float(value.detach()):.3f}",
            metadata={"spec": self.spec.name, "source": "scalar_compat"},
        )

    def validate_intervention(self, action):
        action = self._coerce_action(action)
        for control_spec in self.spec.action_controls:
            value = action.controls.get(control_spec.name)
            if value is None:
                continue
            if control_spec.kind in ("continuous", "binary"):
                scalar = float(torch.as_tensor(value).detach().cpu().reshape(-1)[0])
                if scalar < control_spec.low - 1e-6:
                    return False
                if scalar > control_spec.high + 1e-6:
                    return False
            elif control_spec.kind == "categorical":
                if len(value) != len(control_spec.categories):
                    return False
        return True

    def candidate_interventions(self, state=None):
        affordance = self.action_affordance(state)
        seeds = torch.linspace(-1.5, 1.5, affordance.n_candidates)
        latent_dim = max(
            1,
            sum(len(c.categories) if c.kind == "categorical" else 1
                for c in self.spec.action_controls),
        )
        actions = []
        for seed in seeds:
            latent = torch.zeros(latent_dim)
            latent[0] = seed
            actions.append(self.decode_action(latent))
        return tuple(actions)

    def initial_resource_map(self):
        return {i: 0.0 for i in range(self.entities.n_entities)}

    def initial_knowledge(self, dim=config.K_DIM, dtype=torch.float32):
        values = list(self.spec.initial_knowledge)
        values = values[:dim] + [0.0] * max(0, dim - len(values))
        return torch.tensor(values[:dim], dtype=dtype)

    def initial_tree_state(self, sigma_root=None):
        state = {
            var.name: float(var.init)
            for var in self.spec.state_vars
        }
        state["payoffs"] = self.initial_resource_map()
        state["terminal"] = False
        state["sigma"] = sigma_root
        self._sync_compat_payoffs(state)
        return state

    def is_continue_response(self, response):
        return self._responses[response].continue_branch

    def intent_shift_for_response(self, response):
        return float(self._responses[response].intent_shift)

    def _state_for_ctx(self, state):
        copied = dict(state)
        copied["payoffs"] = dict(state.get("payoffs", {}))
        return copied

    def _ctx(self, state, action, response):
        action = self._coerce_action(action) if action is not None else action
        return {
            "state": state,
            "action": action,
            "response": response,
            "payoffs": state.setdefault("payoffs", self.initial_resource_map()),
            "adapter": self,
        }

    def _sync_compat_payoffs(self, state):
        payoffs = state.setdefault("payoffs", self.initial_resource_map())
        state["payoff_A"] = payoffs.get(self.focal_actor, 0.0)
        state["payoff_B"] = payoffs.get(self.counterpart_actor, 0.0)
        return state

    def transition(self, state, action, response):
        new_state = self._state_for_ctx(state)
        ctx = self._ctx(new_state, action, response)
        transition = self._transitions.get(response)
        if transition is not None:
            for effect in transition.effects:
                effect.apply(ctx)
            if transition.hook is not None:
                transition.hook(ctx)
        self._sync_compat_payoffs(new_state)
        return new_state

    def branch_action_features(self, action, device=None, dtype=torch.float32):
        values = []
        action = self._coerce_action(action)
        for control_spec in self.spec.action_controls:
            value = action.controls.get(control_spec.name, 0.0)
            if control_spec.kind == "categorical":
                values.extend(torch.as_tensor(
                    value, device=device, dtype=dtype).reshape(-1))
            else:
                values.append(torch.as_tensor(
                    value, device=device, dtype=dtype).reshape(()))

        if not values:
            values = [torch.tensor(0.0, device=device, dtype=dtype)]
        flat = torch.stack([
            torch.as_tensor(v, device=device, dtype=dtype).reshape(())
            for v in values[:3]
        ])
        if flat.numel() < 3:
            flat = torch.cat([
                flat,
                torch.zeros(3 - flat.numel(), device=flat.device, dtype=flat.dtype),
            ])
        return flat

    def encode_action_signal(self, action, context=None):
        from interpretation import build_context

        if context is None:
            context = build_context()
        action = self._coerce_action(action)
        device = torch.as_tensor(action.primary_value).device
        dtype = torch.as_tensor(action.primary_value).dtype
        ctx = torch.as_tensor(context, device=device, dtype=dtype)
        features = self.branch_action_features(action, device=ctx.device,
                                               dtype=ctx.dtype)
        return torch.cat([features, ctx], dim=-1)

    def outcome_quality(self, state, role):
        ctx = self._ctx(self._state_for_ctx(state), None, None)
        if self.spec.quality_expr is not None:
            return self.spec.quality_expr.eval(ctx)
        role = self.resolve_role(role)
        return ctx["payoffs"].get(role, 0.0) + 0.2 * state.get("paths_open", 0.0)

    def resolve_outcome(self, action, response, pie=1.0):
        state = self.initial_tree_state()
        if "pie" in state:
            state["pie"] = float(pie)
        next_state = self.transition(state, action, response)
        payoffs = dict(next_state.get("payoffs", {}))
        return Outcome(
            action=action,
            response=response,
            pie_after=float(torch.as_tensor(next_state.get("pie", pie)).detach()),
            paths_open=float(torch.as_tensor(
                next_state.get("paths_open", 0.0)).detach()),
            terminal=bool(next_state.get("terminal", False)),
            raw_state=next_state,
            entity_payoffs=payoffs,
            payoff_A=payoffs.get(self.focal_actor, 0.0),
            payoff_B=payoffs.get(self.counterpart_actor, 0.0),
        )

    def outcome_features(self, outcome, role, device=None, dtype=torch.float32):
        raw_state = dict(outcome.raw_state)
        raw_state.setdefault("payoffs", dict(outcome.entity_payoffs or {}))
        ctx = self._ctx(raw_state, outcome.action, outcome.response)
        values = []
        for feature in self.spec.outcome_features:
            value = feature.expr.eval(ctx)
            values.append(torch.as_tensor(value, device=device, dtype=dtype))
        return torch.stack([v.reshape(()) for v in values])


def with_responses(spec: GameSpec, responses):
    return replace(spec, responses=tuple(responses))
