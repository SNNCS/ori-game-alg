# Application Authoring Guide

This guide explains how to build different applications on top of the current
architecture without escaping the `GameSpec -> GenericGameAdapter ->
CognitiveAgent` boundary.

## Core Rule

Applications normally do not create new adapters.

The only normal way to inject a new game, task, negotiation setting, or
decision environment is:

```text
Application idea
  -> GameSpec
  -> GenericGameAdapter(GameSpec)
  -> CognitiveAgent(adapter=...)
```

There is no game-specific compatibility adapter layer. New games should be
declared as `GameSpec` objects and interpreted by `GenericGameAdapter`. A thin
adapter subclass is only appropriate for external simulator hooks such as
custom observations, dynamic legality, or explicit continuation actions.

For complex applications, use the runtime path on top of the same adapter:

```text
GameSpec
  -> GenericGameAdapter
  -> RuntimeSnapshot
  -> Observation
  -> BeliefState
  -> ActionEvent
  -> WorldResponse
  -> TransitionResult
  -> Trajectory
```

`ActionEvent` is the selected intervention. `WorldResponse` is the observed
world or counterpart event after that intervention. Do not use the selected
action label as the response-prediction target.

## What An Application May Define

An application may define:

- entities and role bindings;
- state variables;
- action controls;
- response/event labels;
- transition effects;
- outcome utility features;
- optional quality expression;
- optional hooks for dynamics that the small DSL cannot express cleanly;
- UI, service, data loading, scenario presets, or training scripts around the
  agent.

An application must not define:

- a new adapter class for ordinary game logic that the DSL can express;
- a new rule class just to compute payoff or legality;
- direct edits to `CognitiveAgent`, `FutureTreeGen`, `BranchPolicy`, or
  `OutcomeUtilityEvaluator` for one application;
- hard-coded latent-vector slices such as `z[4:12]`;
- code that treats raw outcome as reward without going through utility
  features.

## Minimal Application Template

Create a spec module, usually under `specs/`:

```python
from game_spec import (
    AddPayoff, ControlSpec, EntitySpec, FeatureSpec, GameSpec,
    ResponseSpec, RoleBinding, SetTerminal, StateVarSpec, TransitionSpec,
    control, payoff, response_is,
)

move = control("move")

MY_APP_SPEC = GameSpec(
    name="my_app",
    entities=(
        EntitySpec("agent"),
        EntitySpec("counterpart"),
        EntitySpec("observer"),
    ),
    roles=RoleBinding(
        focal="agent",
        counterpart="counterpart",
        observer="observer",
    ),
    state_vars=(
        StateVarSpec("paths_open", init=1.0),
    ),
    action_controls=(
        ControlSpec("move", kind="continuous", low=0.0, high=1.0),
    ),
    responses=(
        ResponseSpec("success", intent_shift=-0.02),
        ResponseSpec("failure", intent_shift=0.04),
    ),
    transitions=(
        TransitionSpec("success", (
            AddPayoff("focal", move),
            AddPayoff("counterpart", 1.0 - move),
            SetTerminal(True),
        )),
        TransitionSpec("failure", (
            AddPayoff("focal", 0.0),
            AddPayoff("counterpart", 0.0),
            SetTerminal(True),
        )),
    ),
    outcome_features=(
        FeatureSpec("self_payoff", payoff("focal")),
        FeatureSpec("other_payoff", payoff("counterpart")),
        FeatureSpec("success", response_is("success")),
    ),
    quality_expr=payoff("focal"),
)
```

Run it through the generic adapter:

```python
from agent import CognitiveAgent
from generic_adapter import GenericGameAdapter
from specs.my_app import MY_APP_SPEC

agent = CognitiveAgent(adapter=GenericGameAdapter(MY_APP_SPEC))
snapshot = agent.runtime_snapshot()
action_event = agent.act(snapshot=snapshot)
```

## Control Types

Use `ControlSpec` to describe what the focal agent can control.

Continuous control:

```python
ControlSpec("price", kind="continuous", low=0.0, high=1.0)
```

Binary control:

```python
ControlSpec("commit", kind="binary", low=0.0, high=1.0)
```

Categorical control:

```python
ControlSpec(
    "message_type",
    kind="categorical",
    categories=("cooperate", "threaten", "delay"),
)
```

`GenericGameAdapter.decode_action(...)` converts generated latent action
vectors into a `GroundedAction` with typed controls. Application code should
inspect `decision.selected.action.controls`, not raw latent vectors.

## Response And Event Design

Use response labels for what the world, opponent, market, group, or simulator
returns after the focal action.

For two-player games, labels can be direct opponent actions:

```text
other_cooperate
other_defect
```

For multi-party applications, use aggregate events in v1:

```text
high_group_contribution
low_group_contribution
market_accepts
market_rejects
```

Do not rewrite `FutureTreeGen` for joint multi-agent branching in an
application. If a multi-agent environment is too rich for aggregate events, add
a spec hook or external simulator hook while keeping the adapter interface
unchanged.

## Transition DSL

Use small effect objects to describe state changes:

```python
TransitionSpec("accept", (
    AddPayoff("focal", payoff_amount),
    AddPayoff("counterpart", other_amount),
    SetState("paths_open", 0.0),
    SetTerminal(True),
))
```

Available effects:

- `SetState(name, value)`
- `AddState(name, value)`
- `ScaleState(name, factor)`
- `AddPayoff(role, value)`
- `SetTerminal(value)`

Available expressions:

- `state("name")`
- `control("name")`
- `payoff("role")`
- `response_is("label")`
- `const(value)`
- arithmetic operators: `+`, `-`, `*`, `/`
- helpers: `expr_abs`, `expr_min`, `expr_max`, `clamp`

Prefer DSL effects over Python hooks when the logic is simple. This keeps the
game inspectable and testable.

## Hook Policy

Hooks are allowed only when the DSL would make the rule unclear.

Acceptable hook uses:

- external simulator step;
- nontrivial auction clearing;
- stochastic market process;
- dynamic legality check that depends on external state;
- complex feature extraction from raw observations.

Unacceptable hook uses:

- choosing the focal agent's action;
- bypassing `CandidateInterventionGenerator`;
- replacing `GenericGameAdapter`;
- directly updating model parameters;
- inventing the next agent action inside a transition hook without exposing it
  through a runtime/continuation contract;
- embedding game-specific logic inside `CognitiveAgent` or `FutureTreeGen`.

Hooks should accept the adapter context and mutate only the transition state
provided by that context.

## Application Patterns

Matrix game:

- binary or categorical action control;
- response labels represent counterpart actions;
- transitions add payoffs and terminate.

Negotiation or bargaining:

- continuous controls for price, concession, delay, commitment;
- categorical controls for message type;
- response labels represent accept, reject, counter, delay;
- use `continue_branch=True` for counter/delay responses.
- provide an explicit simulator or `continuation_action` hook if the planner
  should expand beyond the immediate world response.

Auction:

- continuous control for bid;
- responses such as win/lose or price-band events;
- outcome features should separate self surplus and seller revenue.

Public goods or group game:

- continuous control for focal contribution;
- aggregate response events for group behavior;
- avoid explicit joint branching in v1.

Text-heavy application:

- do not inject free text directly into the core architecture in v1;
- map text choices to categorical controls or outgoing signal features;
- keep language generation/parsing in the application layer.

## Required Tests For Every Application

Add or update tests with the application spec:

```text
1. GenericGameAdapter(spec) instantiates.
2. CognitiveAgent(adapter=...) can act.
3. decode_action returns a valid GroundedAction.
4. transition updates state/payoff/terminal correctly for each response.
5. outcome_features returns finite tensors.
6. response labels drive future-tree child count.
7. runtime transition starts from the provided `RuntimeSnapshot`.
8. response-prediction targets are `WorldResponse`, not `ActionEvent`.
9. private observation fields are masked per viewer.
10. delayed payoff is assigned through a trajectory or terminal target builder.
```

Run:

```powershell
python -B -m unittest discover -s tests -p "test_*.py"
python -B verify_architecture.py
python -B demo.py
```

## Decision Checklist

Before adding an application, answer these questions:

```text
What are the entities?
Which entity is focal?
Which entity is counterpart?
Is there an observer?
What state variables matter?
What can the focal agent control?
What events can the world return?
Which events continue the interaction?
How does each event change state/payoff?
Which outcome features should utility learn from?
Does this require a hook, or can the DSL express it?
```

If the answer involves "create a new adapter," stop and rewrite it as a
`GameSpec` first.
