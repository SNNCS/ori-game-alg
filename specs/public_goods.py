"""Public Goods game with aggregate group-response events."""

from game_spec import (
    AddPayoff, ControlSpec, EntitySpec, FeatureSpec, GameSpec, ResponseSpec,
    RoleBinding, SetTerminal, StateVarSpec, TransitionSpec, control, payoff,
    response_is,
)

contrib = control("contribution")
endowment = 1.0
multiplier = 1.6
n_players = 4.0


def public_goods_return(group_average):
    total = contrib + group_average * (n_players - 1.0)
    return endowment - contrib + multiplier * total / n_players


PUBLIC_GOODS_SPEC = GameSpec(
    name="public_goods",
    entities=(EntitySpec("participant"), EntitySpec("group"), EntitySpec("observer")),
    roles=RoleBinding("participant", "group", "observer"),
    state_vars=(StateVarSpec("paths_open", init=0.0),),
    action_controls=(ControlSpec("contribution", low=0.0, high=1.0),),
    responses=(
        ResponseSpec("high_group_contribution", intent_shift=-0.03),
        ResponseSpec("low_group_contribution", intent_shift=0.03),
    ),
    transitions=(
        TransitionSpec("high_group_contribution", (
            AddPayoff("focal", public_goods_return(0.8)),
            AddPayoff("counterpart", public_goods_return(0.8)),
            SetTerminal(True),
        )),
        TransitionSpec("low_group_contribution", (
            AddPayoff("focal", public_goods_return(0.2)),
            AddPayoff("counterpart", public_goods_return(0.2)),
            SetTerminal(True),
        )),
    ),
    outcome_features=(
        FeatureSpec("self_payoff", payoff("focal")),
        FeatureSpec("group_payoff", payoff("counterpart")),
        FeatureSpec("contribution", contrib),
        FeatureSpec("high_group", response_is("high_group_contribution")),
        FeatureSpec("terminal", 1.0),
    ),
    initial_knowledge=(1.0, 1.0, 0.5),
    quality_expr=payoff("focal"),
)
