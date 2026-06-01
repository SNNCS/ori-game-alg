# Testing Strategy

This project is small enough that the test suite should stay focused on
behavioral contracts rather than broad framework scaffolding. The goal is to
protect the closed-loop Ultimatum Game architecture whenever code changes.

## Test Layers

1. Unit contract tests

   These guard local invariants that should almost never change:

   - config dimensions stay internally consistent;
   - `build_signal` preserves tensor gradients from generated actions;
   - `RelationGraph.G` remains a trainable parameter;
   - relation edges are clamped on read without mutating stored values;
   - the Ultimatum adapter keeps raw outcomes and utility features behind the
     domain boundary.

2. Module and adapter integration tests

   These verify that adjacent modules still compose correctly:

   - latent actions are generated before adapter grounding;
   - generated actions stay inside the adapter affordance;
   - adapter response labels drive `HistorySummarizer` and `BranchPolicy`
     sizes;
   - `DecisionEngine` scores every candidate future and returns normalized
     action probabilities.

3. Closed-loop regression tests

   These protect the architecture claims from `AGENTS.md` and
   `first-principles-ai-architecture.md`:

   - `interpret_and_plan` creates an intent matrix, outgoing signal, and future
     tree;
   - future-tree branch probability mass stays coherent, including after path
     dependence;
   - `act -> observe -> learn` updates mutable episode state while keeping it
     detached;
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
- changing action formation: update latent action and adapter grounding tests;
- changing response labels or adapter semantics: update adapter size and
  outcome tests;
- changing future-tree probabilities: update probability-mass and metric
  differentiability tests;
- changing learning losses: update closed-loop gradient tests;
- changing ablation semantics: update understanding usefulness tests.

Prefer public interfaces such as `CognitiveAgent.act`,
`CognitiveAgent.interpret_and_plan`, `UltimatumGameAdapter`, and
`FutureTreeGen.evaluate`. Avoid tests that depend on private implementation
details unless the detail is one of the explicit architecture invariants.
