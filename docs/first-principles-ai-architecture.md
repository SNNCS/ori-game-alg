# First-Principles AI Architecture

This document defines the target architecture for an agent that can learn
signals, infer intent, and choose actions that improve its own future position.
It is written from first principles and maps back to the current
Ultimatum-Game codebase.

## Status

Draft design with the first implementation slice in place. The current code now
has `CounterfactualPlanner`, `decision.py`, and
`CognitiveAgent.deliberate`, so the agent can score candidate interventions and
choose its own bid. `action_model.py` now adds learned latent action generation:
the agent proposes action vectors and the adapter decodes them into executable
interventions. It also has `experience.py`, `Outcome`,
`OutcomeUtilityEvaluator`, and `LearningSignal`, so raw observed outcomes,
realized utility, and prediction/value/policy losses are distinct in code.
`signal_model.py` now adds learned outgoing communicative signals, and
`FutureTreeGen` conditions world-response prediction on those signals.
`game_spec.py` and `generic_adapter.py` now provide the preferred scenario
extension path: games are declared as Python dataclass specs, then interpreted
by `GenericGameAdapter`. The repo includes six built-in specs: Ultimatum,
Prisoner's Dilemma, Chicken, Stag Hunt, Public Goods, and First-price Auction.
The broader closed loop is now typed end to end for compact declarative specs:
observations produce belief state, candidate actions are interpreted into
intent, world responses train response prediction, and trajectories build
role-relative return targets. Full external simulators and learned joint-policy
rollouts remain extension points rather than hard-coded core behavior.

The runtime-contract slice is now also in place. `runtime.py` defines
`RuntimeSnapshot`, `Observation`, `ActionEvent`, `WorldResponse`,
`TransitionResult`, `TerminalOutcome`, and schema compatibility metadata.
`GenericGameAdapter` exposes wrappers for observation encoding, action-event
grounding, typed world-response transitions, and runtime schema checks.
`belief.py` adds explicit
`BeliefState`, and `trajectory.py` adds trajectory-level learning contracts so
single-step experience is now the length-one case of a broader trajectory
interface.

## Core Claim

The agent does not understand the world for its own sake. It understands the
world because it must act, and bad action has consequences.

The primitive problem is:

```text
Given current situation and current internal model,
which intervention produces the best expected future position?
```

A compact form:

```text
a_t = argmax_a E[FuturePosition | S_t, M_t, do(a)]
```

Where:

- `S_t` is the current objective situation, only partially observable.
- `M_t` is the agent's internal model of the situation, other agents, signals,
  relations, rules, and expected consequences.
- `a` is a candidate intervention. It may be a physical/game action, a
  communicative signal, or both.
- `do(a)` means the agent cares about the counterfactual effect of taking that
  action, not just passively predicting what would happen.

Understanding is therefore not defined as "predicting another agent's mind."
Understanding is defined as:

```text
An internal model M_t is useful understanding iff using M_t improves action
selection and therefore improves expected future position.
```

## Minimal Non-Hardcoded Priors

"No hardcoding" cannot mean "no structure." A system with no prior structure
cannot know what should be learned. The architecture should keep only these
minimal priors:

- Time exists: decisions happen in a sequence.
- The agent can observe: observations carry partial information.
- The agent can intervene: actions change future distributions.
- Other agents may act: future outcomes depend on others' policies.
- Consequences matter: the agent has some utility or future-position function.
- Learning is error correction: failed prediction and failed action both update
  the model.

What must not be hardcoded:

- Fixed semantic slices inside latent vectors, such as `z[4:12] = dignity`.
- Fixed response formulas, such as "unfair offer means reject probability is X."
- Fixed interpretation labels as the only definition of understanding.
- A direct equality between raw outcome and reward.
- A demo script choosing actions by hand while the agent only explains them.
- Game-specific decision logic embedded in general cognition modules.
- Fixed roles, action labels, response counts, or outcome feature names inside
  the core cognition loop.

Domain adapters may still define current action affordances and transitions.
For example, the Ultimatum Game adapter may define that bids live in `[0, 1]`
and responses are `accept`, `reject`, or `counter`. That is not the same as
hardcoding the agent's psychology or policy. The current code pairs that
adapter boundary with a learned candidate-action generator, so the agent now
forms latent actions before they are grounded by the adapter.

## Closed Loop

The target loop is:

```text
S_t                       objective situation, partly hidden
  -> o_t                  observation / incoming signal
  -> M_t                  updated internal model
  -> A_t                  generated candidate interventions
  -> T(a) for each a      counterfactual future simulations
  -> V(T(a))              future-position evaluation
  -> a*                   chosen intervention
  -> environment response actual outcome / new observation
  -> learning updates     world model, belief model, value model, action model
```

This loop separates three things that are often mistakenly collapsed:

```text
outcome          = what actually happened
utility/reward   = how good that outcome and its future consequences are
prediction error = where the agent's model was wrong
```

For example, "I offered 0.8 and the responder accepted" is an outcome. It is
not automatically a reward. Its utility depends on material payoff, future
relationship, retaliation risk, bargaining position, information gained, and
the agent's own goals.

## Target Modules

### 1. Environment Schema

Declares the external domain without deciding for the agent.

Responsibilities:

- Define valid observation spaces.
- Define valid action and signal spaces.
- Decode latent actions into executable interventions.
- Apply objective state transitions.
- Expose legality constraints.
- Return raw outcomes.

It must not:

- Choose the agent's action.
- Generate the agent's latent action vectors.
- Encode fixed psychological meanings.
- Convert outcomes directly into utility.

For the current repo, `GameSpec` plus `GenericGameAdapter` is this boundary.
The default Ultimatum environment is constructed directly through
`GenericGameAdapter(build_ultimatum_spec())`.

### 2. Belief Model

Maintains `M_t`, the agent's internal state.

Responsibilities:

- Update belief from observations and prior actions.
- Store relational memory.
- Store learned signal meanings.
- Represent uncertainty about other agents, rules, and future response patterns.

Current mapping:

- `RelationGraph` is relational memory.
- `InterpretationEngine` is part of belief update.
- `HistorySummarizer` and `sigma` are situation memory.
- `SignalGenerator` uses belief state to emit outgoing communicative signals.

Needed change:

- Treat `Z` as part of a broader belief state, not as the final product.

### 3. Signal Model

Signals are not only inputs. Signals are actions that change what other agents
believe and do.

Responsibilities:

- Interpret incoming signals.
- Generate outgoing signals.
- Predict how outgoing signals alter other agents' future responses.

Examples:

```text
physical action: offer 0.7
communicative signal: "this is my final offer"
joint intervention: offer 0.7 plus explanation
```

Current mapping:

- `build_signal(action)` still encodes the observed physical action.
- `SignalGenerator` proposes a learned outgoing signal vector from the action
  encoding, responder intent, and actor situation.
- `CandidateIntervention.signal` now carries this outgoing signal.

Needed change:

- Generalize this into a public two-way signal interface:
  `encode_observation(...)` and `propose_signal(...)`.

### 4. World / Opponent Model

Predicts what the environment and other agents may do after an intervention.

Responsibilities:

- Estimate `p(o_{t+1}, outcome_t | M_t, do(a))`.
- Predict other agents' responses.
- Predict how relationships and future optionality may change.

Current mapping:

- `FutureTreeGen.policy` is currently closer to an opponent response model than
  to the focal agent's own policy.

Needed change:

- Rename or conceptually treat branch response prediction as world modeling,
  not self-action selection.

### 5. Counterfactual Planner

Uses the world model to simulate futures for candidate interventions.

Responsibilities:

- For each candidate `a`, generate `T(a)`.
- Keep uncertainty and branch probabilities differentiable where possible.
- Produce comparable future distributions.

Current mapping:

- `FutureTreeGen` is the prototype of this module.

Needed change:

- `FutureTreeGen` should be callable as `simulate(action, belief_state)`.
- It should not hide the self-action decision by enumerating all bids inside a
  single tree and leaving final choice outside the agent.

### 6. Future-Position / Utility Model

Evaluates a simulated or observed trajectory.

Responsibilities:

- Convert outcomes and predicted trajectories into utility.
- Learn from experience when simulated value was wrong.
- Keep raw outcome separate from reward.

Future position should be multi-factor:

```text
FuturePosition =
    material payoff
  + bargaining power
  + relationship stability
  + norm legitimacy
  + information advantage
  + future optionality
  - retaliation risk
  - rule violation cost
```

These terms should be learned heads or configurable utility adapters, not
hardcoded formulas inside the planner.

Current mapping:

- `FutureTreeGen.evaluate` currently computes `path_quality`, `risk_floor`, and
  `optionality`. This is a useful start, but it is not a complete utility model.

### 7. Action Policy / Decision Engine

Chooses what the agent actually does.

Responsibilities:

- Generate latent candidate interventions.
- Ground latent actions through current environment affordances.
- Ask the planner to simulate each candidate.
- Use the utility model to score each simulated future.
- Select or sample the final action.

The action policy does not replace the future tree. It uses the future tree.

```text
FutureTree:    "If I do action a, what futures are possible?"
UtilityModel:  "How good are those futures for me?"
ActionPolicy:  "Given those scores, what do I do now?"
```

Current mapping:

- The original `demo.py` handwrote bids with `bid = 0.6 + 0.05 * step`.
- `CandidateInterventionGenerator` now creates latent action vectors from
  actor situation, counterpart situation, relation edge, and context.
- `GenericGameAdapter.decode_action` grounds those vectors into executable
  controls declared by the active `GameSpec`.
- The current demo now calls `CognitiveAgent.deliberate`, which scores
  generated candidate futures and selects a bid/signal intervention.
- `CognitiveAgent.act`, `CognitiveAgent.observe`, and `CognitiveAgent.learn`
  provide the initial public closed-loop interface.

### 8. Learning Coordinator

Applies the correct update to the correct module.

Responsibilities:

- Update belief/signal/world models from prediction errors.
- Update value model from realized future-position error.
- Update action policy from expected utility or regret.
- Preserve end-to-end gradient flow where appropriate.

It must distinguish:

```text
interpretation error: "I misunderstood what the signal implied."
world error:          "I predicted the wrong response or transition."
value error:          "I evaluated the future incorrectly."
policy error:         "I chose poorly despite the available model."
```

Current mapping:

- `experience.py` defines `LearningSignal` with separated response-prediction,
  value, and policy losses.
- `CognitiveAgent.build_experience` creates an `ExperienceStep` from a decision
  and an observed `Outcome`.
- `CognitiveAgent.commit_experience` writes the observed response and payoffs
  into mutable episode state.

## Loss Design

The current code mostly proves that gradients flow through `G`, `I`, and `T`.
The target architecture needs losses that correspond to the closed loop.

### Prediction Loss

Trains the belief, signal, and world models:

```text
L_prediction =
    error(predicted next observation, actual next observation)
  + error(predicted response distribution, actual response)
  + error(predicted state transition, actual transition)
```

### Value Loss

Trains the future-position evaluator:

```text
L_value = error(V_predicted_before_action, U_realized_after_action)
```

`U_realized_after_action` is not the raw outcome. It is the utility model's
assessment of the resulting trajectory or state.

### Policy Objective

Trains the action policy to choose interventions that improve expected future
position:

```text
maximize E[V(T(a))] with exploration and risk constraints
```

or equivalently:

```text
L_policy = -E[FuturePosition]
```

### Understanding Usefulness Loss

This is the key replacement for "understanding equals intent accuracy."

Compare action quality with and without the learned internal model:

```text
usefulness(M_t) =
    FuturePosition(policy using M_t)
  - FuturePosition(policy using ablated/stale/noisy M_t)
```

The model understands more when its internal state improves downstream action.

This can be approximated by:

- counterfactual ablations of `Z`, `G`, history, or signal embeddings;
- regret between chosen action and best action under later-corrected models;
- value improvement from using updated belief versus prior belief.

Current mapping:

- `evaluation.py` defines `AblationSpec` and `UsefulnessReport`.
- `CognitiveAgent.probe_understanding_usefulness` compares full deliberation
  against `NO_UNDERSTANDING`.
- `verify_architecture.py` asserts that ablating understanding changes expected
  position, candidate scores, and soft action probabilities.

## Target Data Interfaces

Future implementation should introduce explicit data objects so concepts do not
collapse into each other:

```text
Observation:
  raw event seen by the agent

BeliefState:
  observation-derived internal model; action-conditioned Z is added only when
  interpreting a concrete candidate action

CandidateIntervention:
  latent action, adapter-decoded physical/game action, optional signal

PredictedFuture:
  tree or rollout distribution conditioned on do(action)

Outcome:
  what actually happened after action

Utility:
  evaluated future-position value of outcome/trajectory

LearningSignal:
  prediction error, value error, and policy credit assignment
```

Current implementation note:

```text
ActionEvent:
  what the focal actor did

WorldResponse:
  what the environment or other agents did after that action

Trajectory:
  ordered ActionEvent / WorldResponse / TransitionResult steps
```

Prediction targets should be `WorldResponse` objects. Passing an
`ActionEvent` or a raw response label as a response-prediction target is a type
error.

## Migration Plan From Current Code

### Step 1: Split "simulate one action" from "enumerate all actions"

Status: implemented.

Refactor `FutureTreeGen` so the counterfactual planner can build a future tree
for one candidate action event:

```text
CounterfactualPlanner.simulate(belief_state, action_event, snapshot) -> FutureTree
```

Keep a separate helper for enumerating all legal candidates. This makes the
future tree a planner primitive instead of a hidden chooser.

### Step 2: Add `DecisionEngine`

Status: implemented as an initial utility-scored selector in `decision.py`.

Create a module that:

1. receives generated or supplied candidate interventions;
2. calls `CounterfactualPlanner.simulate` for each candidate;
3. evaluates each tree;
4. returns a selected action plus diagnostics.

This removes handwritten demo actions.

### Step 3: Introduce explicit `Outcome` and `Utility`

Status: implemented as an initial interface. `FuturePositionEvaluator`
separates planner metrics from predicted utility, while `Outcome`,
`OutcomeUtilityEvaluator`, and `RealizedUtility` separate observed outcomes from
realized utility.

Stop treating `path_quality` or raw response as reward. Add a utility interface
that evaluates observed and simulated outcomes.

### Step 4: Add learned outgoing signals

Status: implemented as an initial continuous signal vector.

`SignalGenerator` emits an `OutgoingSignal` with no fixed semantic labels.
`FutureTreeGen.policy` consumes it when predicting responses, so signal meaning
is learned through prediction/value/policy gradients rather than hardcoded
message categories.

### Step 5: Add an experience loop

Status: implemented as a typed runtime loop. The demo now performs
`runtime_snapshot -> act -> transition_event -> trajectory -> learn`, and
`learn` commits trajectory-derived experience.

The agent should expose:

```text
observe(o_t)
deliberate()
act()
learn(outcome)
```

Runtime observation, belief update, and counterfactual planning now represent
the full closed loop; the old single-action `interpret_and_plan` entrypoint has
been removed.

### Step 6: Train understanding by action improvement

Status: partially implemented as an ablation probe.

Add ablation or regret checks that answer:

```text
Does using the learned belief state produce better action choices than ignoring
it?
```

This makes "understanding" operational rather than decorative.

The current verification checks sensitivity rather than final superiority:
removing understanding changes expected position and policy scores. Stronger
future work should compare realized future position over multiple environments
or opponent policies.

### Step 7: Move domain discreteness behind specs/adapters

Status: implemented as a generic spec/adapter boundary.

`GameSpec` now owns concrete entity roles, action controls, response labels,
continue-branch semantics, initial public knowledge, state transitions, raw
outcome resolution effects, and outcome utility features. `GenericGameAdapter`
interprets these specs and exposes the stable contract consumed by core modules
such as `CognitiveAgent`, `HistorySummarizer`, `BranchPolicy`,
`FutureTreeGen`, and `OutcomeUtilityEvaluator`.

This solves the immediate hardcoding problem for roles/responses/features and
moves action grounding behind declarative affordances rather than fixed core
action lists.

### Step 8: Generate latent actions before grounding

Status: implemented as an initial differentiable action generator.

`CandidateInterventionGenerator` combines:

```text
actor_sigma || counterpart_sigma || G[actor, counterpart] || context
```

through learned matrices plus learnable action slots to produce latent action
vectors. The adapter decodes each vector into an executable action. In the
current Ultimatum Game, the first latent coordinate is decoded into a continuous
kept-share bid within the adapter's affordance range.

This is still not fully open-ended action invention: only one physical control
dimension is grounded today. The next step is multi-field grounding, where a
latent action can decode into resource movement, timing, information release,
commitment strength, and communicative signal controls.

## Success Criteria

The architecture is not complete until these are true:

- The agent chooses its own action; demo actions are not handwritten.
- Candidate actions can be generated as latent vectors, then grounded through
  adapter affordances.
- The chosen action depends on the learned internal model.
- The chosen intervention includes a learned outgoing signal.
- Future tree generation is conditioned on candidate interventions.
- Outcome, utility, and prediction error are represented separately.
  Initial implementation: `Outcome`, `RealizedUtility`, and `LearningSignal`.
- The model can update both "what I believe" and "how I act" from experience.
- No latent vector is interpreted through fixed semantic slices.
- No raw game outcome is treated as reward without a utility interface.
- No core cognition module assumes fixed role IDs, action lists, response
  counts, or outcome feature names; role and response facts belong to adapters,
  while action formation belongs to the generator plus adapter grounding.
- A verification script can show that changing or ablating the belief model
  changes expected future position and the action-value landscape.

## One-Sentence Summary

The current system can interpret and imagine; the target system must observe,
update belief, simulate counterfactual interventions, evaluate future position,
choose an action, observe the result, and learn separately from prediction
error and action-value error.
