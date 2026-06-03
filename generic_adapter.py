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
from runtime import (
    ActionEvent, Observation, ObservationSpec, RuntimeSchema,
    RuntimeSnapshot, TerminalOutcome, TransitionResult, WorldResponse,
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

    def counterpart_for_actor(self, actor):
        actor = int(actor)
        if actor == self.focal_actor:
            return self.counterpart_actor
        if actor == self.counterpart_actor:
            return self.focal_actor
        raise ValueError(
            f"No default counterpart declared for actor {actor}. "
            "Override counterpart_for_actor/response_actor for this spec.")

    def response_actor(self, snapshot=None, action_event=None,
                       world_response=None, actor=None):
        if isinstance(world_response, WorldResponse):
            if world_response.source is not None:
                return int(world_response.source)
        if isinstance(action_event, ActionEvent):
            return self.counterpart_for_actor(action_event.actor)
        if actor is not None:
            return self.counterpart_for_actor(actor)
        if isinstance(snapshot, RuntimeSnapshot) and snapshot.current_actor is not None:
            return self.counterpart_for_actor(snapshot.current_actor)
        return self.counterpart_actor

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

    def runtime_schema(self):
        return RuntimeSchema(
            spec_name=self.spec.name,
            spec_version=getattr(self.spec, "schema_version", "1"),
            observation_features=tuple(var.name for var in self.spec.state_vars),
            action_controls=tuple(control.name
                                  for control in self.spec.action_controls),
            world_response_labels=self.response_labels(),
            outcome_features=self.outcome_feature_names,
            n_entities=self.entities.n_entities,
        )

    def action_affordance(self, state=None, actor=None):
        primary = self.spec.action_controls[0]
        if isinstance(state, RuntimeSnapshot):
            state = state.state
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
            metadata={"spec": self.spec.name, "source": "scalar"},
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

    def initial_runtime_snapshot(self, sigma_root=None, current_actor=None):
        state = self.initial_tree_state(sigma_root)
        return RuntimeSnapshot(
            state=state,
            current_actor=(
                self.focal_actor if current_actor is None else current_actor),
            terminal=bool(state.get("terminal", False)),
            public={
                key: value for key, value in state.items()
                if key not in ("payoffs", "sigma")
            },
            private={},
            metadata={"spec": self.spec.name},
        )

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
        self._ensure_payoffs(state)
        return state

    def is_continue_response(self, response):
        response = self._response_label(response)
        return self._responses[response].continue_branch

    def intent_shift_for_response(self, response):
        response = self._response_label(response)
        return float(self._responses[response].intent_shift)

    def _response_label(self, response):
        if isinstance(response, WorldResponse):
            return response.label
        return response

    def _state_for_ctx(self, state):
        copied = dict(state)
        copied["payoffs"] = dict(state.get("payoffs", {}))
        return copied

    def _ctx(self, state, action, response):
        action = self._coerce_action(action) if action is not None else action
        response = self._response_label(response)
        return {
            "state": state,
            "action": action,
            "response": response,
            "payoffs": state.setdefault("payoffs", self.initial_resource_map()),
            "adapter": self,
        }

    def _ensure_payoffs(self, state):
        state.setdefault("payoffs", self.initial_resource_map())
        return state

    def transition(self, state, action, response):
        new_state = self._state_for_ctx(state)
        ctx = self._ctx(new_state, action, response)
        response = self._response_label(response)
        transition = self._transitions.get(response)
        if transition is not None:
            for effect in transition.effects:
                effect.apply(ctx)
            if transition.hook is not None:
                transition.hook(ctx)
        self._ensure_payoffs(new_state)
        return new_state

    def continuation_action(self, state, previous_action, previous_response):
        """Optional simulator hook for deterministic continuation actions.

        The generic core does not invent the next agent's strategy. Rich
        applications can override this hook or provide an external simulator
        adapter; the default keeps continuation branches as leaves for future
        value estimation.
        """
        return None

    # ----- runtime wrappers ----------------------------------------------

    def _public_observation_items(self, snapshot):
        if not isinstance(snapshot, RuntimeSnapshot):
            snapshot = RuntimeSnapshot(state=snapshot)
        state_var_names = {var.name for var in self.spec.state_vars}
        names = []
        values = []
        for var in self.spec.state_vars:
            names.append(var.name)
            values.append(snapshot.state.get(
                var.name, snapshot.public.get(var.name, var.init)))
        public_source = (
            snapshot.public if snapshot.public else {
                key: value for key, value in snapshot.state.items()
                if key not in ("payoffs", "sigma")
            }
        )
        for key in sorted(public_source):
            if key not in state_var_names and key not in ("payoffs", "sigma"):
                names.append(key)
                values.append(public_source[key])
        return tuple(names), values

    def _private_observation_items(self, snapshot, viewer):
        if not isinstance(snapshot, RuntimeSnapshot):
            snapshot = RuntimeSnapshot(state=snapshot)
        private_state = snapshot.private.get(int(viewer), {})
        names = tuple(sorted(private_state))
        values = [private_state[key] for key in names]
        return names, values

    def _flatten_feature_names(self, names, values, dim=None):
        flat_names: list[str] = []
        for name, value in zip(names, values):
            width = int(self._flatten_numeric(value).numel())
            if width == 0:
                continue
            if width == 1:
                flat_names.append(str(name))
            else:
                flat_names.extend(f"{name}[{idx}]" for idx in range(width))
        if dim is not None:
            flat_names = flat_names[:dim]
        return tuple(flat_names)

    def observation_spec(self, snapshot=None, viewer=None, dim=config.K_DIM):
        if snapshot is None:
            public_names = tuple(var.name for var in self.spec.state_vars)
            private_names = ()
        else:
            if viewer is None:
                viewer = self.focal_actor
            public_base_names, public_values = self._public_observation_items(
                snapshot)
            private_base_names, private_values = self._private_observation_items(
                snapshot, viewer)
            public_names = self._flatten_feature_names(
                public_base_names, public_values, dim=dim)
            remaining = (
                None if dim is None else max(int(dim) - len(public_names), 0))
            private_names = self._flatten_feature_names(
                private_base_names, private_values, dim=remaining)
        return ObservationSpec(
            feature_names=public_names,
            private_feature_names=private_names,
            dim=dim,
            schema_version=getattr(self.spec, "schema_version", "1"),
        )

    @staticmethod
    def _flatten_numeric(value, device=None, dtype=torch.float32):
        if isinstance(value, bool):
            value = float(value)
        if isinstance(value, (int, float)) or torch.is_tensor(value):
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
            return tensor.reshape(-1)
        return torch.zeros(0, device=device, dtype=dtype)

    def _pack_features(self, values, dim=config.K_DIM, device=None,
                       dtype=torch.float32, pad=True):
        chunks = [
            self._flatten_numeric(value, device=device, dtype=dtype)
            for value in values
        ]
        numeric_chunks = [chunk for chunk in chunks if chunk.numel() > 0]
        if numeric_chunks:
            vec = torch.cat(numeric_chunks)
        else:
            vec = torch.zeros(0, device=device, dtype=dtype)
        if dim is not None:
            vec = vec[:dim]
        mask = torch.ones(vec.numel(), device=vec.device, dtype=dtype)
        if pad and dim is not None and vec.numel() < dim:
            pad = torch.zeros(dim - vec.numel(), device=vec.device, dtype=dtype)
            vec = torch.cat([vec, pad])
            mask = torch.cat([
                mask,
                torch.zeros(dim - mask.numel(), device=vec.device, dtype=dtype),
            ])
        return vec, mask

    def encode_public_state(self, snapshot, dim=config.K_DIM,
                            device=None, dtype=torch.float32):
        _, values = self._public_observation_items(snapshot)
        return self._pack_features(values, dim=dim, device=device, dtype=dtype)

    def encode_private_state(self, snapshot, viewer, dim=config.K_DIM,
                             device=None, dtype=torch.float32):
        _, values = self._private_observation_items(snapshot, viewer)
        return self._pack_features(values, dim=dim, device=device, dtype=dtype)

    def encode_observation(self, snapshot, viewer=None, dim=config.K_DIM,
                           device=None, dtype=torch.float32):
        if not isinstance(snapshot, RuntimeSnapshot):
            snapshot = RuntimeSnapshot(state=snapshot)
        if viewer is None:
            viewer = self.focal_actor
        _, public_values = self._public_observation_items(snapshot)
        _, private_values = self._private_observation_items(snapshot, viewer)
        public_vec, public_mask = self._pack_features(
            public_values, dim=None, device=device, dtype=dtype, pad=False)
        private_vec, private_mask = self._pack_features(
            private_values, dim=None, device=public_vec.device,
            dtype=public_vec.dtype, pad=False)
        vector = torch.cat([public_vec, private_vec])[:dim]
        mask = torch.cat([public_mask, private_mask])[:dim]
        if vector.numel() < dim:
            pad = torch.zeros(
                dim - vector.numel(), device=vector.device,
                dtype=vector.dtype)
            vector = torch.cat([vector, pad])
            mask = torch.cat([mask, torch.zeros_like(pad)])
        spec = self.observation_spec(snapshot=snapshot, viewer=viewer, dim=dim)
        return Observation(
            viewer=int(viewer),
            vector=vector,
            mask=mask,
            spec=spec,
            public_state=dict(snapshot.public),
            private_state=dict(snapshot.private.get(int(viewer), {})),
            snapshot_step=int(snapshot.step_index),
        )

    def legal_action_mask(self, snapshot=None, actor=None):
        if snapshot is None:
            return None
        if not isinstance(snapshot, RuntimeSnapshot):
            return None
        actor = self.focal_actor if actor is None else int(actor)
        mask = snapshot.legal_action_mask.get(actor)
        if mask is None:
            return None
        return torch.as_tensor(mask, dtype=torch.bool)

    def ground_action(self, action_or_policy, snapshot=None, actor=None):
        actor = self.focal_actor if actor is None else int(actor)
        log_prob = getattr(action_or_policy, "log_prob", None)
        action = getattr(action_or_policy, "grounded_action", None)
        if action is None:
            action = action_or_policy
        action = self._coerce_action(action)
        return ActionEvent(
            actor=actor,
            action=action,
            label=action.display,
            controls=dict(action.controls),
            log_prob=log_prob,
            metadata={"spec": self.spec.name},
        )

    def transition_event(self, snapshot, action_event, world_response):
        if not isinstance(action_event, ActionEvent):
            raise TypeError("transition_event requires an ActionEvent.")
        if not isinstance(world_response, WorldResponse):
            raise TypeError("transition_event requires a WorldResponse.")
        before = snapshot if isinstance(snapshot, RuntimeSnapshot) else RuntimeSnapshot(
            state=snapshot)
        next_state = self.transition(
            before.state, action_event.action, world_response)
        after = RuntimeSnapshot(
            state=next_state,
            current_actor=before.current_actor,
            step_index=before.step_index + 1,
            terminal=bool(next_state.get("terminal", False)),
            public={
                key: value for key, value in next_state.items()
                if key not in ("payoffs", "sigma")
            },
            private=before.private,
            legal_action_mask=before.legal_action_mask,
            metadata=dict(before.metadata),
        )
        terminal = None
        if after.terminal:
            terminal = TerminalOutcome(
                snapshot=after,
                payoffs=dict(next_state.get("payoffs", {})),
                terminal=True,
                metadata={"world_response": world_response.label},
            )
        return TransitionResult(
            before=before,
            action_event=action_event,
            world_response=world_response,
            after=after,
            terminal_outcome=terminal,
        )

    def outcome_from_transition(self, transition_result):
        state = transition_result.after.state
        payoffs = dict(state.get("payoffs", {}))
        features = {
            key: value for key, value in state.items()
            if key not in ("payoffs", "sigma")
        }
        return Outcome(
            action=transition_result.action_event.action,
            world_response=transition_result.world_response,
            terminal=bool(state.get("terminal", False)),
            raw_state=state,
            entity_payoffs=payoffs,
            features=features,
        )

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
        return ctx["payoffs"].get(role, 0.0)

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
