"""Chicken game as an aggregate-response GameSpec."""

from game_spec import (
    AddPayoff, ControlSpec, EntitySpec, FeatureSpec, GameSpec, ResponseSpec,
    RoleBinding, SetTerminal, StateVarSpec, TransitionSpec, control, payoff,
    response_is,
)

swerve = control("swerve")

CHICKEN_SPEC = GameSpec(
    name="chicken",
    entities=(EntitySpec("driver"), EntitySpec("other"), EntitySpec("observer")),
    roles=RoleBinding("driver", "other", "observer"),
    state_vars=(StateVarSpec("paths_open", init=0.0),),
    action_controls=(ControlSpec("swerve", kind="binary", low=0.0, high=1.0),),
    responses=(
        ResponseSpec("other_swerve", intent_shift=-0.03),
        ResponseSpec("other_straight", intent_shift=0.05),
    ),
    transitions=(
        TransitionSpec("other_swerve", (
            AddPayoff("focal", swerve * 2.0 + (1.0 - swerve) * 4.0),
            AddPayoff("counterpart", swerve * 2.0),
            SetTerminal(True),
        )),
        TransitionSpec("other_straight", (
            AddPayoff("focal", swerve * 1.0 + (1.0 - swerve) * -5.0),
            AddPayoff("counterpart", swerve * 4.0 + (1.0 - swerve) * -5.0),
            SetTerminal(True),
        )),
    ),
    outcome_features=(
        FeatureSpec("self_payoff", payoff("focal")),
        FeatureSpec("other_payoff", payoff("counterpart")),
        FeatureSpec("swerve", swerve),
        FeatureSpec("other_swerve", response_is("other_swerve")),
        FeatureSpec("terminal", 1.0),
    ),
    initial_knowledge=(1.0, 0.4, 0.6),
    quality_expr=payoff("focal"),
)
