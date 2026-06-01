"""Stag Hunt as an aggregate-response GameSpec."""

from game_spec import (
    AddPayoff, ControlSpec, EntitySpec, FeatureSpec, GameSpec, ResponseSpec,
    RoleBinding, SetTerminal, StateVarSpec, TransitionSpec, control, payoff,
    response_is,
)

stag = control("hunt_stag")

STAG_HUNT_SPEC = GameSpec(
    name="stag_hunt",
    entities=(EntitySpec("hunter"), EntitySpec("partner"), EntitySpec("observer")),
    roles=RoleBinding("hunter", "partner", "observer"),
    state_vars=(StateVarSpec("paths_open", init=0.0),),
    action_controls=(ControlSpec("hunt_stag", kind="binary", low=0.0, high=1.0),),
    responses=(
        ResponseSpec("partner_stag", intent_shift=-0.04),
        ResponseSpec("partner_hare", intent_shift=0.03),
    ),
    transitions=(
        TransitionSpec("partner_stag", (
            AddPayoff("focal", stag * 4.0 + (1.0 - stag) * 3.0),
            AddPayoff("counterpart", stag * 4.0),
            SetTerminal(True),
        )),
        TransitionSpec("partner_hare", (
            AddPayoff("focal", (1.0 - stag) * 3.0),
            AddPayoff("counterpart", stag * 3.0 + (1.0 - stag) * 3.0),
            SetTerminal(True),
        )),
    ),
    outcome_features=(
        FeatureSpec("self_payoff", payoff("focal")),
        FeatureSpec("other_payoff", payoff("counterpart")),
        FeatureSpec("stag_commitment", stag),
        FeatureSpec("partner_stag", response_is("partner_stag")),
        FeatureSpec("terminal", 1.0),
    ),
    initial_knowledge=(1.0, 0.7, 0.3),
    quality_expr=payoff("focal"),
)
