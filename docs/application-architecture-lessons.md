# Application Architecture Lessons

This note captures practical lessons from applying the architecture to a
deployable game product. It is written as a review checklist for future
applications that need to stay faithful to the core model instead of becoming
a collection of scripts around it.

## Architectural Boundary

The normal application path is:

```text
GameSpec
  -> GenericGameAdapter(GameSpec)
  -> CognitiveAgent
  -> environment/runtime observation
  -> ExperienceStep
  -> learning signal
```

Applications may define a game world, UI, persistence, training loop, and
external simulator hooks. They should not define a custom adapter for ordinary
domain logic, and they should not put game-specific action policy inside
`CognitiveAgent`, `FutureTreeGen`, `BranchPolicy`, or utility evaluators.

The adapter is the contract boundary. If a game needs dynamic legality,
role-dependent outcome features, or an external rules runtime, add generic
hooks to `GameSpec` and interpret them from `GenericGameAdapter`. Do not create
a new game-specific adapter class unless the generic contract itself is wrong.

## Discrete Facts vs Learnable Paths

Some parts of an application are necessarily discrete:

- card dealing, legal actions, turn order, all-in rules, side pots, and
  showdown in poker;
- auction clearing;
- public-good group aggregation;
- state persistence and restore;
- UI labels and localized display text.

Those facts belong in the environment/runtime or transition hook.

The following should remain tensorized and learnable:

- observation and action-signal encoding;
- latent action generation;
- categorical action preferences;
- continuous amount preferences;
- response probabilities;
- future-position value;
- realized utility;
- prediction, value, utility, and policy losses.

A good rule of thumb: the environment may say what is legal and what happened;
the model should decide what to try and learn how likely or valuable the
consequences are.

## Response Semantics

Use response labels for what the world or other agents do after the focal
action. Do not train response prediction on the focal action itself.

For example, in a multi-party poker-like environment:

```text
ActionEvent:    focal actor folded, called, raised, or went all-in
WorldResponse:  others folded, called around, faced raise, street advanced,
                showdown win, showdown loss, split
```

`ActionEvent` is a runtime fact used to advance the simulator. `WorldResponse`
is the prediction target for `P(response | belief, do(action))`. Mixing these
two concepts turns the prediction loss into an echo of the chosen action and
breaks the closed loop.

For multi-agent games, v1 applications may use aggregate world responses rather
than full joint branching. That is acceptable as long as the labels still mean
"what the world did next", not "what I just selected".

## Avoiding Strategy Hardcoding

Hardcoding can enter through several quiet paths:

- fallback bots that choose raise/call/fold by fixed probabilities;
- product previews that compute branch probabilities with hand-written
  formulas;
- training opponents that use thresholds while only the main agent learns;
- checkpoint compatibility filters that silently leave critical heads random;
- scripted counterfactual rollouts that are mistaken for learned policy.

Allowed deterministic code:

- legality masks;
- minimum and maximum action amounts;
- simulator rules;
- serialization, restore, and display formatting;
- mapping internal labels to user-facing text.

Not allowed as strategy:

- "raise with probability 0.18";
- "call if cost <= 12";
- "all-in only when stack is short";
- "fold risk = 0.25 + aggression * 0.25";
- "z[4:12] means dignity".

If a fallback is required for development, it should be explicitly labeled as
not part of training or evaluation. Production and training gates should fail
when a required model is missing or incompatible instead of silently switching
to a scripted policy.

## Product-Layer Interpretation

Applications often need to show users what the agent is "thinking". That
display must be a projection of model outputs, not a separate rule engine.

Good sources for product explanations:

- selected action and action probability from the decision result;
- root branch probabilities from the selected future tree;
- future metrics such as path quality, risk floor, and optionality;
- learned utility features and weights;
- observed world-response history.

Unsafe sources:

- hand-written formulas over public state that pretend to be model beliefs;
- fixed personality text unrelated to the model state;
- direct semantic claims about latent vector slices.

Explanations should say what can be justified by the projection:

```text
The model's most likely branch is "faced raise" at 41%.
The selected candidate is "bet/raise" with action probability 36%.
The future-position evaluator gives path_quality +0.18 and risk_floor -0.04.
```

They should not claim hidden mental concepts unless those concepts are measured
by an explicit learned head or feature.

## Semantic Correctness Checklist

Before training or deployment, verify these questions:

1. Are action labels, world-response labels, and UI labels separate concepts?
2. Does response prediction train on observed world responses rather than the
   selected action?
3. Are raw outcomes, realized utility, prediction loss, value loss, utility
   loss, and policy loss kept separate?
4. Does every role-relative payoff or feature receive the acting role, not a
   fixed focal/counterpart pair?
5. Does the future tree use the current environment state rather than a fresh
   initial state?
6. Do counterfactual branches produce distinct states when their responses
   differ?
7. Are legality masks used only to constrain invalid actions, not to encode
   strategic preferences?
8. Do checkpoint loaders reject or clearly report missing critical heads?
9. Does evaluation fail on model errors rather than hiding them in counters?
10. Can a reviewer trace each user-visible probability back to a model output?

## Training Loop Checklist

A faithful training loop should collect trajectories at the unit where payoff
is actually resolved. For poker-like games this means the whole hand, not a
single immediate event.

For each decision step:

1. generate latent candidate actions through `CandidateInterventionGenerator`;
2. decode them through `GenericGameAdapter` using current legality;
3. use `CognitiveAgent.act` to select a grounded action;
4. apply the action to the environment/runtime;
5. observe a later `WorldResponse`;
6. at terminal or payoff resolution, build role-relative outcomes;
7. compute prediction, value, utility, and policy losses;
8. commit experience to mutable episode state.

Do not pass a complete hand-written candidate set into `agent.act` as the
default training path. That is useful for focused tests, but it bypasses the
learned action generator in normal training.

## Deployment Gates

Deployment should verify more than "the service starts":

- required checkpoints exist;
- checkpoint tensor shapes match all critical modules;
- action generator, branch policy, utility heads, and history encoder load;
- inference can produce a legal action for every active seat;
- product projections can produce branch previews from model futures;
- environment restore can continue a game without changing hidden facts;
- evaluation exits nonzero or is marked failed when model errors are nonzero.

If the application changed response labels, outcome features, or action
controls, old checkpoints should be treated as stale unless compatibility is
explicitly proven. Shape-filtered partial loading is useful for migration, but
it must not be reported as a fully trained deployed model.

## Review Heuristics

When reviewing an application built on this architecture, search for:

```text
heuristic
fallback
random
threshold
if ... raise/call/fold
response_for_action
z[
.item()
float(...)
except Exception: pass
```

These terms are not always wrong, but they are high-signal places where an
application may have escaped the architecture. The reviewer should classify
each occurrence as one of:

- environment fact;
- legality constraint;
- display formatting;
- test fixture;
- product projection;
- strategy policy.

Only the last category is a violation.

## First-Principles Summary

An application is aligned when the rules say what can happen, the model learns
what to do, and the product layer faithfully projects what the model computed.

It is misaligned when rules choose actions, UI formulas pretend to be beliefs,
or training labels describe the agent's own action rather than the world's
response.

