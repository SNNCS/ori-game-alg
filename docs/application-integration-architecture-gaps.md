# Application Integration Architecture Gaps

This note explains why the current first-principles game-cognition
architecture is promising, but difficult to apply directly to complex product
environments without adding extra application-side machinery. It is written
from the perspective of integrating the architecture into real games, services,
training pipelines, and deployable products.

The short version:

```text
GameSpec -> GenericGameAdapter -> CognitiveAgent
```

is the right boundary for simple and medium-complexity games. The gap is that
the current codebase still behaves like a compact research prototype. It proves
that action generation, interpretation, future simulation, utility evaluation,
and learning can be wired together, but it does not yet provide all of the
contracts needed for stable training in rich, partially observable,
multi-agent environments.

## What Works Well

The architecture has several strong ideas that should be preserved.

`GameSpec` is a good declarative environment boundary. It keeps game facts,
roles, controls, responses, transitions, and outcome features outside the core
cognitive modules.

`GenericGameAdapter` is the right contract layer. Applications should not write
a new adapter for every game. They should define environment facts and let the
adapter expose a uniform interface to `CognitiveAgent`.

The separation between raw outcome, realized utility, prediction error, value
error, and policy error is correct. The architecture explicitly rejects the
common mistake that "what happened" is automatically "reward."

The latent vector invariants are also correct. The model should not assign
hard-coded semantic slices such as `z[4:12] = dignity`. Product explanations
should be projections of model outputs, not fixed meanings attached to latent
coordinates.

The main issue is not that the architecture points in the wrong direction. The
issue is that several essential application contracts are still too thin.

## Gap 1: The Learning Coordinator Is Not Yet a Full Training System

The current architecture has `Outcome`, `OutcomeUtilityEvaluator`, and
`LearningSignal`, but the actual learning interface is still closer to a
single-step research loop than to a production training coordinator.

Complex games need learning at the unit where payoff is actually resolved. In
poker this is a whole hand, sometimes a whole table session. In negotiation it
may be a multi-turn bargain. In markets it may be a full clearing cycle.

The architecture does not yet define first-class support for:

- trajectory buffers;
- terminal return assignment;
- role-relative Monte Carlo return;
- temporal-difference targets;
- bootstrap value targets;
- advantage estimation;
- replay or curriculum;
- evaluation-gated checkpoint selection.

Because these pieces are missing, application code must invent them around the
agent. That is where many semantic mistakes appear: training on immediate
events instead of resolved payoff, using the selected action as the response
target, or training a planner score without training the continuation value it
depends on.

Recommended architectural change:

```text
ExperienceStep[]
  -> Trajectory
  -> ReturnBuilder
  -> PredictionTarget / ValueTarget / PolicyTarget
  -> LearningCoordinator
```

This should live in the architecture, not in each product script.

## Gap 2: FutureTreeGen Mixes World Model, Planner, and Value Surface

`FutureTreeGen` currently plays several roles:

- it predicts response branch probabilities;
- it expands counterfactual futures;
- it applies transition semantics through the adapter;
- it computes metrics such as `path_quality`, `risk_floor`, and `optionality`;
- it may include learned bootstrap value at non-terminal leaves.

These are related but not identical concepts.

In a simple game, this compression is fine. In a complex game, it creates
confusion:

```text
Branch probability: P(world response | belief, do(action))
Planner tree:       possible future states under candidate action
Utility model:      how good those future states are
Policy objective:   how to choose among candidate actions
```

When these are not separated, it becomes unclear which module should receive
which training signal. For example, a depth-1 planner with learned bootstrap
value is architecturally sound only if the bootstrap critic has its own target.
If the critic only receives weak indirect gradients through action scores, it
is connected in code but not actually learning the continuation value.

Recommended architectural change:

```text
WorldResponseModel: P(response | belief, do(action))
CounterfactualPlanner: build T(action, belief_state)
FutureValueModel: V(state, role)
DecisionEngine: choose/sample action from scored futures
```

`FutureTreeGen` can remain the implementation, but its interfaces should make
these responsibilities explicit.

## Gap 3: GameSpec Needs a Stronger Environment Runtime Contract

`GameSpec` works well for matrix games, bargaining games, small auctions, and
other environments where transitions are compact. For richer simulations,
applications need dynamic runtime facts:

- current legal actions;
- role-relative observations;
- hidden information masks;
- current acting entity;
- min/max amount bounds;
- simulator snapshot and restore;
- terminal versus non-terminal payoff semantics;
- distinct action events and world responses.

The current hook mechanism is useful, but too open-ended. Hooks can express
complex dynamics, yet the architecture does not strongly type what a hook must
provide. This lets product code accidentally mix environment facts, training
labels, product display, and policy behavior.

Recommended architectural change:

```text
RuntimeSnapshot
ObservationSpec(viewer)
ActionAffordance(state)
ActionEvent
WorldResponse
TransitionResult
TerminalOutcome
```

These should be generic types, not product-local conventions.

## Gap 4: ActionEvent and WorldResponse Are Not Type-Protected

The architecture says response labels should describe what the world or other
agents do after the focal action. That is correct.

However, code can still accidentally do this:

```text
selected action: raise
training response label: raised
```

That trains the response model to echo the agent's own action, rather than to
predict the world response to `do(action)`.

The correct separation is:

```text
ActionEvent
  what the acting agent did to the runtime:
  folded, called, raised, went all-in

WorldResponse
  what the environment or other agents did after that action:
  others folded, called around, faced raise, street advanced,
  showdown win/loss/split
```

This distinction is important enough that it should be part of the architecture
type system. If it remains an application convention, complex products will
eventually mix the labels.

Recommended architectural change:

- `TransitionSpec` and runtime hooks should consume `ActionEvent`.
- branch prediction and response loss should train on `WorldResponse`.
- tests should fail when a response target equals the selected action event by
  construction.

## Gap 5: Multi-Agent Games Are Represented, Not Fully Modeled

The architecture has multiple agents, relation graph rows, role embeddings,
and an intent matrix. That is useful.

But the future tree is still closer to:

```text
focal action -> aggregate response
```

than to:

```text
seat_i action -> seat_j action -> seat_k action -> ... -> terminal outcome
```

The documentation accepts aggregate responses for multi-agent v1. That is a
reasonable first step, but it becomes lossy for games with:

- strict turn order;
- changing position;
- hidden information;
- multiple opponents with different stacks or incentives;
- side pots or other path-dependent settlement;
- multiple future decision points before payoff resolution.

Aggregate responses are useful, but they should be clearly marked as an
approximation. Applications should not mistake an aggregate response model for
a complete joint multi-agent planner.

Recommended architectural change:

- keep aggregate response branching for v1;
- add a later `JointTurnModel` or `SequentialResponseModel`;
- let applications choose between aggregate and explicit sequential branching.

## Gap 6: Observation Encoding Is Not a First-Class Contract

The current application pattern often pushes environment facts into a context
vector. That is not enough for partially observable games.

Complex environments need a generic way to declare and test:

- what the acting role can observe;
- what is public;
- what is private;
- what must be masked;
- how observations are encoded into tensors;
- whether a saved checkpoint expects a specific observation layout.

Without a first-class observation contract, applications can accidentally leak
hidden state, omit role-relative features, or change feature order without
checkpoint invalidation.

Recommended architectural change:

```text
GameSpec.observation_features
adapter.encode_observation(state, viewer)
adapter.encode_public_state(state)
adapter.encode_private_state(state, viewer)
```

The model should learn from observations, not from raw simulator snapshots.

## Gap 7: Mixed Action Spaces Need Stronger Policy Support

The current latent candidate generator works well for simple continuous
actions. Complex games often require mixed actions:

```text
categorical action kind: fold / check-call / bet-raise / all-in
continuous amount:      raise size or bet fraction
binary flags:           commitment, reveal, pressure, etc.
```

The generic adapter can decode continuous, binary, and categorical controls,
but the policy and training interface do not yet treat mixed action output as a
first-class object.

Applications need:

- legal categorical masks;
- sampled training and argmax deployment;
- action log-probabilities;
- continuous amount distributions;
- gradient-preserving repeated amount candidates;
- coverage checks for every legal action kind.

If this remains hidden inside adapter hooks and slot latents, applications can
accidentally create legal coverage with dead continuous gradients, or use fixed
candidate anchors that look learnable but are not.

Recommended architectural change:

```text
ActionPolicyOutput:
  candidate_logits
  legal_mask
  continuous_params
  selected_index
  log_prob
  grounded_action
```

This should still decode through `GenericGameAdapter`; it should not be a
game-specific policy.

## Gap 8: Utility Targets Need Stronger Semantic Rules

The architecture correctly separates raw outcome from utility, but applications
still need strict rules for target construction.

Common failure modes:

- using raw stack change as payoff before an interaction is terminal;
- treating live committed resources as realized loss;
- normalizing payoff by a scale so large that the learning signal disappears;
- evaluating all roles from the focal actor's perspective;
- training value on immediate event payoff when final payoff is delayed.

These are not strategy mistakes. They are semantic target mistakes. They make a
faithful model learn the wrong objective.

Recommended architectural change:

```text
OutcomeTargetBuilder(role, trajectory)
  -> realized_target
  -> continuation_target
  -> utility_features
  -> normalization metadata
```

Every target should be role-relative and should declare its scale.

## Gap 9: Checkpoint Compatibility Is an Architecture Concern

In real products, action controls, response labels, outcome features, and model
heads change over time. A checkpoint may load partially or fail silently if the
application is not strict.

The architecture should treat checkpoints as part of the environment contract.

Needed metadata:

- spec name and version;
- action control schema;
- response label schema;
- observation feature schema;
- outcome feature schema;
- candidate count;
- model head shapes;
- training status and evaluation summary.

If any of these differ, the loader should clearly report the mismatch and the
deployment gate should fail. It should not silently fall back to random heads or
scripted behavior.

## Gap 10: Deployment Evaluation Is Not Optional

Research verification answers:

```text
Do gradients flow?
Does snapshot -> act -> transition -> trajectory -> learn run?
Does ablation change candidate scores?
```

Product verification must also answer:

```text
Does every active role produce legal actions?
Is the action distribution degenerate?
Are all-in or terminal actions exploding?
Are some action kinds never used?
Does the game complete?
Are world responses diverse and sensible?
Is payoff variance dominated by one broken seat?
Can the service restore and continue a saved state?
```

Without deployment gates, a training run can save a final checkpoint that is
technically compatible and behaviorally unusable. Evaluation gates should not
change training loss, but they should decide which checkpoint can be deployed.

Recommended architectural change:

```text
Training run
  -> periodic evaluation
  -> gate report
  -> best checkpoint selection
  -> deployment eligibility
```

This belongs near the architecture, not only in product scripts.

## Why Applications Escape the Architecture

Applications usually escape the architecture for practical reasons:

1. The environment has facts that `GameSpec` cannot express cleanly.
2. The training loop needs trajectory-level credit assignment.
3. The product needs user-facing explanations immediately.
4. The runtime needs legal actions and persistence.
5. The model has not learned enough yet, so developers add heuristic fallbacks.

Each reason is understandable. But each can turn into hardcoding unless the
architecture provides the correct extension point.

The right fix is not to forbid application code. The right fix is to move
repeated application-side patterns into generic architectural contracts.

## Recommended Roadmap

Priority 1: typed environment runtime contract.

Add generic types for observation, action affordance, action event, world
response, transition result, terminal outcome, and simulator snapshot.

Initial implementation status: `runtime.py` now provides these dataclasses,
default schema compatibility checks, and adapter wrappers through
`GenericGameAdapter`.

Priority 2: trajectory learning coordinator.

Add first-class support for multi-step trajectories, role-relative returns,
bootstrap targets, and separated prediction/value/policy updates.

Initial implementation status: `trajectory.py` now provides trajectory steps,
return/target builders, and a coordinator that preserves existing single-step
learning as the length-one case.

Priority 3: mixed action policy output.

Represent categorical logits, legal masks, continuous amount distributions, and
log-probabilities directly, while still grounding through the adapter.

Priority 4: value and utility target builders.

Make payoff scale, terminal semantics, live commitment semantics, and
role-relative targets explicit.

Priority 5: evaluation and checkpoint gate.

Make deployment health checks a standard part of training, not an optional
application script.

Priority 6: product projection contract.

Expose model-derived explanation primitives: selected action probability,
branch probabilities, future metrics, utility feature weights, and recent
world-response history.

## Final Assessment

The architecture is not fundamentally blocked. It is strong enough to support
real applications if the application remains disciplined.

The main weakness is that the current codebase leaves too many high-stakes
semantics to product scripts:

- what the agent observed;
- what action was legal;
- what the world did in response;
- when payoff was actually realized;
- how continuation value was trained;
- which checkpoint was deployable.

For simple games, this is manageable. For complex games, these missing
contracts cause developers to patch behavior in hooks, runners, UI projections,
or training scripts. That is how hardcoding enters even when the core
architecture is principled.

The next architectural step should be to turn the repeated application lessons
into formal contracts, so product code can stay thin and the model can learn
inside the intended loop.
