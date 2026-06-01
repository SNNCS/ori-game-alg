"""Prisoner's Dilemma as an aggregate-response GameSpec."""

from game_spec import (
    AddPayoff, ControlSpec, EntitySpec, FeatureSpec, GameSpec, ResponseSpec,
    RoleBinding, SetTerminal, StateVarSpec, TransitionSpec, control, payoff,
    response_is,
)

c = control("cooperate")

PRISONERS_DILEMMA_SPEC = GameSpec(
    name="prisoners_dilemma",
    entities=(EntitySpec("player"), EntitySpec("other"), EntitySpec("observer")),
    roles=RoleBinding("player", "other", "observer"),
    state_vars=(StateVarSpec("paths_open", init=0.0),),
    action_controls=(ControlSpec("cooperate", kind="binary", low=0.0, high=1.0),),
    responses=(
        ResponseSpec("other_cooperate", intent_shift=-0.02),
        ResponseSpec("other_defect", intent_shift=0.04),
    ),
    transitions=(
        TransitionSpec("other_cooperate", (
            AddPayoff("focal", c * 3.0 + (1.0 - c) * 5.0),
            AddPayoff("counterpart", c * 3.0),
            SetTerminal(True),
        )),
        TransitionSpec("other_defect", (
            AddPayoff("focal", (1.0 - c) * 1.0),
            AddPayoff("counterpart", c * 5.0 + (1.0 - c) * 1.0),
            SetTerminal(True),
        )),
    ),
    outcome_features=(
        FeatureSpec("self_payoff", payoff("focal")),
        FeatureSpec("other_payoff", payoff("counterpart")),
        FeatureSpec("cooperation", c),
        FeatureSpec("other_cooperated", response_is("other_cooperate")),
        FeatureSpec("terminal", response_is("other_cooperate") + response_is("other_defect")),
    ),
    initial_knowledge=(1.0, 0.5, 0.5),
    quality_expr=payoff("focal"),
)
