"""First-price auction as a GameSpec."""

from game_spec import (
    AddPayoff, ControlSpec, EntitySpec, FeatureSpec, GameSpec, ResponseSpec,
    RoleBinding, SetTerminal, StateVarSpec, TransitionSpec, control, payoff,
    response_is, state,
)

bid = control("bid")

FIRST_PRICE_AUCTION_SPEC = GameSpec(
    name="first_price_auction",
    entities=(EntitySpec("bidder"), EntitySpec("seller"), EntitySpec("observer")),
    roles=RoleBinding("bidder", "seller", "observer"),
    state_vars=(
        StateVarSpec("value", init=1.0),
        StateVarSpec("paths_open", init=0.0),
    ),
    action_controls=(ControlSpec("bid", low=0.0, high=1.0),),
    responses=(
        ResponseSpec("win", intent_shift=-0.02),
        ResponseSpec("lose", intent_shift=0.02),
    ),
    transitions=(
        TransitionSpec("win", (
            AddPayoff("focal", state("value") - bid),
            AddPayoff("counterpart", bid),
            SetTerminal(True),
        )),
        TransitionSpec("lose", (
            AddPayoff("focal", 0.0),
            AddPayoff("counterpart", 0.0),
            SetTerminal(True),
        )),
    ),
    outcome_features=(
        FeatureSpec("self_payoff", payoff("focal")),
        FeatureSpec("seller_revenue", payoff("counterpart")),
        FeatureSpec("bid", bid),
        FeatureSpec("won", response_is("win")),
        FeatureSpec("terminal", 1.0),
    ),
    initial_knowledge=(1.0, 0.5, 0.5),
    quality_expr=payoff("focal"),
)
