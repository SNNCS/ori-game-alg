# Testing Strategy

This project is small enough that the test suite should stay focused on
behavioral contracts rather than broad framework scaffolding. The goal is to
protect the closed-loop generic game architecture whenever code changes.
Ultimatum remains the default smoke-test spec, but the benchmark specs and
runtime contracts are part of the same quality bar.

Tests cannot prove the absence of every semantic error or hard-coded shortcut.
They should, however, make architecture concepts executable: when a concept
boundary is violated, a contract test should fail before the bug reaches a
demo or application.

## Semantic Contract Requirements

Every new test that touches architecture behavior must name the semantic
contract it protects. Prefer tests that would fail if the implementation were
quietly specialized to Ultimatum, poker, fixed role IDs, fixed response labels,
or one-step fully observable games.

Good semantic tests usually include at least one of these pressures:

- a non-Ultimatum spec with different state names, response labels, roles, or
  action controls;
- a role order or viewer change that would reveal fixed focal/counterpart
  assumptions;
- private observation fields that must not leak across viewers;
- a delayed terminal payoff that must not be replaced by an immediate event;
- a dynamic legal-action mask that must change policy probabilities;
- an explicit negative case where the wrong runtime type is rejected.

Avoid tests that only lock down implementation shape. For example, do not test
that a private helper was called; test that `ActionEvent` cannot be used as a
`WorldResponse`, that illegal actions receive zero probability, or that a
planner starts from the supplied `RuntimeSnapshot`.

## Hardcoding Gates

When adding or changing core architecture code, include tests or static checks
that would catch forbidden game-specific assumptions:

- no core loss should depend on selected action labels as rewards or response
  targets;
- no core runtime path should rely on state names such as `pie`, `paths_open`,
  or two-player payoff aliases;
- no core planner should assume response labels such as `accept`, `reject`, or
  `counter`;
- no core model should slice `z` into fixed semantic regions;
- no default planner path should invent another agent's continuation action
  without an adapter or simulator hook;
- no observation encoder should combine public and private state in a way that
  hides the boundary declared by `ObservationSpec`.

If a test needs a concrete example, create a small toy `GameSpec` inside the
test. The toy spec should be intentionally different from Ultimatum in at
least one important dimension so it can expose accidental hardcoding.

## Test Layers

1. Unit contract tests

   These guard local invariants that should almost never change:

   - config dimensions stay internally consistent;
   - `build_signal` preserves tensor gradients from generated actions;
   - `RelationGraph.G` remains a trainable parameter;
   - relation edges are clamped on read without mutating stored values;
   - the generic adapter keeps raw outcomes and utility features behind the
     runtime/spec boundary;
   - runtime dataclasses reject concept mixing, such as using an `ActionEvent`
     as a response-prediction target.

2. Module and adapter integration tests

   These verify that adjacent modules still compose correctly:

   - latent actions are generated before adapter grounding;
   - generated actions stay inside the adapter affordance;
   - adapter response labels drive `HistorySummarizer` and `BranchPolicy`
     sizes;
   - `DecisionEngine` scores every candidate future and returns normalized
     action probabilities;
   - observation specs keep public and private fields viewer-relative;
   - mixed action policy output respects dynamic legal masks.

3. Closed-loop regression tests

   These protect the architecture claims from `AGENTS.md` and
   `first-principles-ai-architecture.md`:

   - runtime planning creates an intent matrix, outgoing signal, and future
     tree from a `RuntimeSnapshot`;
   - future-tree branch probability mass stays coherent, including after path
     dependence;
   - `act -> transition_event -> trajectory -> learn` updates mutable episode
     state while keeping it detached;
   - trajectory learning uses role-relative terminal or continuation targets
     instead of raw events as rewards;
   - the training loss backpropagates through `G`, `I`, `T.policy`,
     `action_gen`, `signal_gen`, and the future-position evaluator;
   - ablating understanding changes the action-value landscape.

## Local Quality Gate

Run this before handing off code:

```powershell
python -B -m unittest discover -s tests -p "test_*.py"
python -B verify_architecture.py
python -B demo.py
```

`python -B` avoids bytecode writes and keeps the checks friendlier on Windows
workspaces where `__pycache__` writes may be blocked.

## CI Quality Gate

GitHub Actions runs the same quality gate on every push and pull request:

1. install `requirements-dev.txt`;
2. run the `unittest` suite;
3. run `verify_architecture.py`;
4. run `demo.py`.

The unit/integration tests should catch narrow contract regressions quickly.
The two scripts remain as executable architecture probes because they print
useful diagnostics for gradient flow, generated actions, outgoing signals, and
understanding ablations.

## When To Add Tests

Add or update tests with the same change that alters behavior:

- changing dimensions: update config tests and any affected module contracts;
- changing action formation: update latent action, legal mask, log-probability,
  and adapter grounding tests;
- changing response labels or adapter semantics: update adapter size and
  outcome tests, including at least one non-default label set;
- changing future-tree probabilities: update probability-mass and metric
  differentiability tests, plus a check that planning starts from the current
  `RuntimeSnapshot`;
- changing learning losses: update closed-loop gradient tests and trajectory
  target tests;
- changing observation encoding: update viewer-relative public/private mask
  tests;
- changing runtime dataclasses: update type-separation tests for
  `ActionEvent`, `WorldResponse`, `Outcome`, and `TerminalOutcome`;
- changing ablation semantics: update understanding usefulness tests.

Prefer public interfaces such as `CognitiveAgent.act`,
`CognitiveAgent.deliberate`, `GenericGameAdapter`, runtime transitions, and
`FutureTreeGen.evaluate`. Avoid tests that depend on private implementation
details unless the detail is one of the explicit architecture invariants.

Before accepting a new architecture test, ask:

- What semantic boundary would this fail on?
- Would it fail if the core code were secretly hard-coded to Ultimatum?
- Would it fail if public/private observation, action/response, or
  outcome/reward concepts were mixed?
- Does it exercise behavior through the public runtime or adapter interface?
