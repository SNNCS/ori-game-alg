"""Ultimatum Game expressed as a GameSpec."""

from game_spec import (
    AddPayoff, AddState, ControlSpec, EntitySpec, FeatureSpec, GameSpec,
    ResponseSpec, RoleBinding, ScaleState, SetState, SetTerminal,
    StateVarSpec, TransitionSpec, const, control, expr_abs, payoff,
    response_is, state,
)


def build_ultimatum_spec(bids=(0.5, 0.6, 0.7, 0.8, 0.9),
                         responses=("accept", "reject", "counter"),
                         outside=0.0,
                         counter_discount=0.9,
                         paths_open_decay=0.1):
    low, high = min(bids), max(bids)
    response_specs = []
    for label in responses:
        response_specs.append(ResponseSpec(
            label=label,
            continue_branch=(label == "counter"),
            intent_shift=(0.05 if label == "reject"
                          else -0.05 if label == "accept"
                          else 0.0),
        ))

    return GameSpec(
        name="ultimatum",
        entities=(EntitySpec("proposer"),
                  EntitySpec("responder"),
                  EntitySpec("observer")),
        roles=RoleBinding(
            focal="proposer",
            counterpart="responder",
            observer="observer",
        ),
        state_vars=(
            StateVarSpec("pie", init=1.0),
            StateVarSpec("paths_open", init=1.0),
        ),
        action_controls=(
            ControlSpec("kept_share", low=low, high=high),
        ),
        responses=tuple(response_specs),
        transitions=(
            TransitionSpec("accept", (
                AddPayoff("focal", state("pie") * control("kept_share")),
                AddPayoff("counterpart",
                          state("pie") * (1.0 - control("kept_share"))),
                SetState("paths_open", 0.0),
                SetTerminal(True),
            )),
            TransitionSpec("reject", (
                AddPayoff("focal", outside),
                AddPayoff("counterpart", outside),
                SetState("paths_open", 0.0),
                SetTerminal(True),
            )),
            TransitionSpec("counter", (
                ScaleState("pie", counter_discount),
                AddState("paths_open", -paths_open_decay),
                SetTerminal(False),
            )),
        ),
        outcome_features=(
            FeatureSpec("self_payoff", payoff("focal")),
            FeatureSpec("other_payoff", payoff("counterpart")),
            FeatureSpec("paths_open", state("paths_open")),
            FeatureSpec("terminal", state("terminal")),
            FeatureSpec("accepted", response_is("accept")),
            FeatureSpec("rejected", response_is("reject")),
            FeatureSpec("countered", response_is("counter")),
            FeatureSpec(
                "fairness_deviation",
                2.0 * expr_abs(control("kept_share") - 0.5),
            ),
        ),
        n_candidates=len(bids),
        initial_knowledge=(1.0, 1.0, 1.0, 0.6, 0.7, 0.7, 0.7, 0.42),
        quality_expr=payoff("focal") + 0.2 * state("paths_open"),
    )


ULTIMATUM_SPEC = build_ultimatum_spec()
