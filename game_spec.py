"""Declarative game specifications and a tiny expression/effect DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch


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


@dataclass(frozen=True)
class GroundedAction:
    """Adapter-decoded action controls ready for environment simulation."""

    controls: Mapping[str, Any]
    primary_value: Any
    display: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __float__(self):
        value = torch.as_tensor(self.primary_value).detach().cpu()
        return float(value.reshape(-1)[0])


@dataclass(frozen=True)
class EntitySpec:
    name: str


@dataclass(frozen=True)
class RoleBinding:
    focal: str
    counterpart: str
    observer: str | None = None


@dataclass(frozen=True)
class StateVarSpec:
    name: str
    init: float = 0.0
    kind: str = "continuous"


@dataclass(frozen=True)
class ControlSpec:
    name: str
    kind: str = "continuous"
    low: float = 0.0
    high: float = 1.0
    categories: Sequence[str] = ()


@dataclass(frozen=True)
class ResponseSpec:
    label: str
    continue_branch: bool = False
    intent_shift: float = 0.0


class Expr:
    def eval(self, ctx):
        raise NotImplementedError

    def __add__(self, other):
        return BinaryExpr(self, as_expr(other), "add")

    def __radd__(self, other):
        return BinaryExpr(as_expr(other), self, "add")

    def __sub__(self, other):
        return BinaryExpr(self, as_expr(other), "sub")

    def __rsub__(self, other):
        return BinaryExpr(as_expr(other), self, "sub")

    def __mul__(self, other):
        return BinaryExpr(self, as_expr(other), "mul")

    def __rmul__(self, other):
        return BinaryExpr(as_expr(other), self, "mul")

    def __truediv__(self, other):
        return BinaryExpr(self, as_expr(other), "div")

    def __rtruediv__(self, other):
        return BinaryExpr(as_expr(other), self, "div")

    def __neg__(self):
        return ConstExpr(0.0) - self


@dataclass(frozen=True)
class ConstExpr(Expr):
    value: Any

    def eval(self, ctx):
        return self.value


@dataclass(frozen=True)
class StateExpr(Expr):
    name: str

    def eval(self, ctx):
        return ctx["state"].get(self.name, 0.0)


@dataclass(frozen=True)
class ControlExpr(Expr):
    name: str

    def eval(self, ctx):
        action = ctx["action"]
        if isinstance(action, GroundedAction):
            return action.controls[self.name]
        if self.name in ("action", "primary", "primary_value"):
            return action
        raise KeyError(self.name)


@dataclass(frozen=True)
class PayoffExpr(Expr):
    role: str | int

    def eval(self, ctx):
        role = ctx["adapter"].resolve_role(self.role)
        return ctx["payoffs"].get(role, 0.0)


@dataclass(frozen=True)
class ResponseIsExpr(Expr):
    label: str

    def eval(self, ctx):
        return 1.0 if ctx["response"] == self.label else 0.0


@dataclass(frozen=True)
class BinaryExpr(Expr):
    left: Expr
    right: Expr
    op: str

    def eval(self, ctx):
        left = self.left.eval(ctx)
        right = self.right.eval(ctx)
        if self.op == "add":
            return left + right
        if self.op == "sub":
            return left - right
        if self.op == "mul":
            return left * right
        if self.op == "div":
            return left / right
        raise ValueError(self.op)


@dataclass(frozen=True)
class UnaryExpr(Expr):
    expr: Expr
    op: str

    def eval(self, ctx):
        value = self.expr.eval(ctx)
        if self.op == "abs":
            return torch.abs(value) if torch.is_tensor(value) else abs(value)
        raise ValueError(self.op)


@dataclass(frozen=True)
class ClampExpr(Expr):
    expr: Expr
    low: float
    high: float

    def eval(self, ctx):
        value = self.expr.eval(ctx)
        if torch.is_tensor(value):
            return torch.clamp(value, self.low, self.high)
        return min(max(value, self.low), self.high)


@dataclass(frozen=True)
class MinExpr(Expr):
    left: Expr
    right: Expr

    def eval(self, ctx):
        left = self.left.eval(ctx)
        right = self.right.eval(ctx)
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.minimum(torch.as_tensor(left), torch.as_tensor(right))
        return min(left, right)


@dataclass(frozen=True)
class MaxExpr(Expr):
    left: Expr
    right: Expr

    def eval(self, ctx):
        left = self.left.eval(ctx)
        right = self.right.eval(ctx)
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.maximum(torch.as_tensor(left), torch.as_tensor(right))
        return max(left, right)


def as_expr(value):
    return value if isinstance(value, Expr) else ConstExpr(value)


def const(value):
    return ConstExpr(value)


def state(name):
    return StateExpr(name)


def control(name):
    return ControlExpr(name)


def payoff(role):
    return PayoffExpr(role)


def response_is(label):
    return ResponseIsExpr(label)


def expr_abs(value):
    return UnaryExpr(as_expr(value), "abs")


def expr_min(left, right):
    return MinExpr(as_expr(left), as_expr(right))


def expr_max(left, right):
    return MaxExpr(as_expr(left), as_expr(right))


def clamp(value, low, high):
    return ClampExpr(as_expr(value), low, high)


class Effect:
    def apply(self, ctx):
        raise NotImplementedError


@dataclass(frozen=True)
class SetState(Effect):
    name: str
    value: Expr | Any

    def apply(self, ctx):
        ctx["state"][self.name] = as_expr(self.value).eval(ctx)


@dataclass(frozen=True)
class AddState(Effect):
    name: str
    value: Expr | Any

    def apply(self, ctx):
        ctx["state"][self.name] = (
            ctx["state"].get(self.name, 0.0) + as_expr(self.value).eval(ctx))


@dataclass(frozen=True)
class ScaleState(Effect):
    name: str
    factor: Expr | Any

    def apply(self, ctx):
        ctx["state"][self.name] = (
            ctx["state"].get(self.name, 0.0) * as_expr(self.factor).eval(ctx))


@dataclass(frozen=True)
class AddPayoff(Effect):
    role: str | int
    value: Expr | Any

    def apply(self, ctx):
        role = ctx["adapter"].resolve_role(self.role)
        ctx["payoffs"][role] = (
            ctx["payoffs"].get(role, 0.0) + as_expr(self.value).eval(ctx))


@dataclass(frozen=True)
class SetTerminal(Effect):
    value: Expr | Any = True

    def apply(self, ctx):
        ctx["state"]["terminal"] = as_expr(self.value).eval(ctx)


@dataclass(frozen=True)
class TransitionSpec:
    response: str
    effects: Sequence[Effect] = ()
    hook: Callable[[dict], None] | None = None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    expr: Expr | Any


@dataclass(frozen=True)
class GameSpec:
    name: str
    entities: Sequence[EntitySpec]
    roles: RoleBinding
    state_vars: Sequence[StateVarSpec]
    action_controls: Sequence[ControlSpec]
    responses: Sequence[ResponseSpec]
    transitions: Sequence[TransitionSpec]
    outcome_features: Sequence[FeatureSpec]
    n_candidates: int = 5
    initial_knowledge: Sequence[float] = ()
    r_public_first: float = 0.5
    quality_expr: Expr | None = None
    schema_version: str = "1"
