# AGENTS.md

This repository is a compact PyTorch implementation of an original
three-structure cognitive architecture for the Ultimatum Game. Use this file as
the first orientation point for future agents. The README contains a concise
user-facing overview; this document is the working map for making safe changes.

## Suggested Skills

- `zoom-out`: use first when you need to rebuild the project map or explain how
  a local change affects the whole architecture.
- `diagnose`: use for broken gradient flow, failing demos, numerical issues, or
  behavior that diverges from the intended game dynamics.
- `tdd`: use when adding new mechanics or losses. Start with focused regression
  tests or a demo assertion before editing the model.
- `handoff`: use before ending a long session so the next agent knows what was
  read, changed, verified, and left open.

## Project Purpose

The project models how agents interpret an observed ultimatum-game action, how
directed relational asymmetry changes that interpretation, and how possible
future outcomes are reconstructed from the resulting intent representation.

The core architecture is:

```text
G: RelationGraph        directed asymmetric relation tensor
I: InterpretationEngine action + relation + rule + situation -> intent matrix Z
T: FutureTreeGen        future outcome tree generated from Z
```

Everything important is a `torch.nn.Module`. A single optimizer over
`CognitiveAgent.parameters()` is expected to train the relation graph, the
interpretation engine, rule stances, tolerance head, latent action generator,
signal generator, utility heads, and future-tree branch policy together through
autograd.

## Non-Negotiable Design Invariants

- Keep the repository specific to the Ultimatum Game. Do not reintroduce
  e-commerce concepts, recommendation flows, discount actions, purchase
  outcomes, or LLM prompt/prefix machinery.
- `G` must remain a trainable `nn.Parameter`. Do not replace it with manual
  edge-update code or a detached table.
- Clamp relation edges on read, not by in-place mutation during forward passes.
  This preserves autograd.
- The intent vector `z` is a learned representation. Do not assign hard-coded
  semantic slices such as `z[4:12]` for "dignity" or similar concepts.
- Future-tree branch probabilities must consume the full responder intent
  vector `z_B`, plus explicit scalar game features such as the offer gap and
  optional tolerance. Avoid hand-coded response formulas over `z` slices.
- Keep branch probabilities as tensors until after loss computation. Avoid
  `.item()`, `float(...)`, or detached conversions in differentiable paths.
- `r_j` rule interpretation and `sigma_j` situation context are part of the
  interpretation input. Do not simplify `I` back to only `[s || edge]`.
- Cognitive dissonance is action-signal based: `BayesianInverse(s, z_j)` creates
  `z*`; there is no discrete text classifier or response-type classifier.
- Mutable episode state (`h`, `omega`, `K`) is not model parameter state.
  History updates intentionally detach so later optimizer steps do not backprop
  through previous episodes.

## File Map

| File | Responsibility |
| --- | --- |
| `config.py` | Global constants, dimensions, hyperparameters, and sanity checks. |
| `game_rule.py` | Ultimatum Game payoff and legality rules. |
| `relation_graph.py` | Structure `G`: trainable directed relation tensor. |
| `situation.py` | Role embeddings, history summaries, resource state, public knowledge, and `sigma` assembly. |
| `interpretation.py` | Structure `I`: signal construction, intent inference, propagation, rule stance, Bayesian inverse, tolerance head, dissonance loss. |
| `game_adapter.py` | Domain adapter for concrete roles/entities, action affordances/decoding, response labels, transitions, outcomes, and utility features. |
| `action_model.py` | Learnable latent action generation before adapter grounding. |
| `future_tree.py` | Structure `T`: branch policy, recursive future tree, path dependence, and evaluation metrics. |
| `signal_model.py` | Learnable outgoing communicative signal generation. |
| `decision.py` | Candidate interventions, predicted futures, future-position scoring, and action selection. |
| `experience.py` | Raw outcomes, realized utility, and separated prediction/value/policy learning signals. |
| `evaluation.py` | Ablation specs and usefulness reports for understanding-as-action-improvement probes. |
| `agent.py` | Composition layer wiring all structures into `CognitiveAgent`. |
| `demo.py` | End-to-end smoke test and gradient-flow verification. |
| `verify_architecture.py` | Lightweight verification for act/observe/learn, outgoing signals, and ablation sensitivity. |
| `__init__.py` | Public exports for the package-style API. |
| `README.md` | User-facing overview and minimal example. |
| `docs/first-principles-ai-architecture.md` | Target architecture for closing the loop from signal understanding to self-beneficial action. |

## Core Domain Vocabulary

- **Focal actor / proposer in the current adapter**:
  `UltimatumGameAdapter.focal_actor`, backed by `config.ACTOR_A == 0`.
  Makes an offer and keeps the `bid` share of the pie.
- **Counterpart / responder in the current adapter**:
  `UltimatumGameAdapter.counterpart_actor`, backed by `config.ACTOR_B == 1`.
  Responds with adapter-provided labels such as `accept`, `reject`, or
  `counter`.
- **Observer in the current adapter**:
  `UltimatumGameAdapter.observer_actor`, backed by `config.ACTOR_C == 2`.
  Receives propagated interpretation from the counterpart.
- **Signal `s`**: deterministic action encoding from `build_signal`.
  Shape is `M=8`: `[bid, 1-bid, fairness_dev] + context(5)`.
- **Relation edge `G[j,i,:]`**: observer `j`'s directed relational view of actor
  `i`, shape `K=32`.
- **Rule stance `r_j`**: learnable per-agent rule-interpretation vector,
  shape `P=16`, regularized toward `UltimatumRule.r_public`.
- **Situation `sigma_j`**: concatenation of role embedding, history summary,
  resource vector, and public knowledge, shape `SIGMA_DIM=40`.
- **Intent matrix `Z`**: shape `(n_agents, D)`, where each row `z_j` is an
  observer's interpretation of an action. The actor's own row is zeroed in
  `compute_Z`.
- **Future tree**: recursively generated state tree over bids and responses.
  It evaluates optionality, risk floor, and expected path quality.

## Important Dimensions

From `config.py`:

```text
N_AGENTS = 3
N_ROLES  = 3
K        = 32    relation-edge vector dimension
D        = 32    intent vector dimension
P        = 16    rule-interpretation dimension
RHO_DIM  = 8     role embedding
H_DIM    = 16    history summary
OMEGA_DIM = 8    resource state
K_DIM    = 8     public knowledge
SIGMA_DIM = 40   role + history + resource + public knowledge
M         = 8    action signal
ACTION_LATENT_DIM = 16
N_GENERATED_ACTIONS = 5
INPUT_DIM = 96   signal + edge + rule stance + sigma
DEPTH     = 2    default future-tree depth
LR        = 1e-3
```

## End-To-End Runtime Flow

The canonical public entry point is `CognitiveAgent.act` in `agent.py`.

1. `CandidateInterventionGenerator` reads actor/counterpart situations,
   relation edge, and context to generate latent action vectors.
2. The adapter decodes each latent action into an executable intervention.
3. `build_signal(action, context)` creates `s` for each grounded action.
4. `CognitiveAgent` builds `r_dict` from each `RuleInterpretation.r_j`.
5. `CognitiveAgent` builds `sigma_dict` via `sigma_of(j)`.
6. `InterpretationEngine.compute_Z` computes every observer row of `Z`.
7. `InterpretationEngine.propagate` computes B-to-C re-signalling as `z_C`.
8. `SignalGenerator` emits a learned outgoing communicative signal for the
   candidate intervention.
9. `FutureTreeGen.simulate_action` asks the adapter for response labels and
   transitions, then reconstructs `T(intervention, signal)` from `Z`.
10. `FutureTreeGen.apply_path_dep` reweights root branches from recent history
   and scales descendant joint probabilities consistently.
11. `FutureTreeGen.evaluate` returns differentiable metrics:
   `optionality`, `risk_floor`, and `path_quality`.
12. `DecisionEngine` scores predicted futures and chooses an action/signal.
13. `experience.py` resolves the observed outcome, evaluates realized utility,
    and separates prediction, value, and policy losses.
14. Training currently combines:

```text
loss = dissonance + 0.01 * rule_reg + experience.learning_signal.total_loss
```

## Module Notes

### `relation_graph.py`

`RelationGraph.G` has shape `(n_agents, n_agents, K)`. The diagonal is
initialized to zero, representing self-view. `get_edge` and `get_row` return
read-time clamped tensors. Diagnostics include `edge_variance` and
`asymmetry(i, j)`.

### `interpretation.py`

`build_context` packs five floats:

```text
turn_pos, session_len, prev_reject_rate, status_gap, urgency
```

`build_signal` treats the physical action as the signal. There is no utterance
text path. It preserves tensor actions so generated latent actions can receive
gradient through downstream action-conditioned prediction. `InterpretationEngine.infer_intent` computes:

```text
z_j = tanh(W_z [s || G[j,i,:] || r_j || sigma_j])
```

`BayesianInverse` maps the real action signal into an intent-correction target
`z*`. `ToleranceHead` reads the whole `z_j`, not a semantic slice.

### `future_tree.py`

`BranchPolicy` learns `P(response | z_B, action, outgoing_signal, tolerance)`
over the adapter-provided response labels. It no longer has a fixed output
size of three in the core layer.

`FutureTreeGen.simulate_action` builds a future tree conditioned on one
candidate intervention. Response labels, transition rules, continue-branch
semantics, and outcome quality come from the adapter.
`FutureTreeGen.generate` remains as the legacy helper that expands all adapter
candidates under one root. `evaluate` normalizes leaf probability mass before
computing tree metrics.

### `game_adapter.py`

`UltimatumGameAdapter` contains the toy-game-specific pieces that should not
belong to the core agent: A/B/C entity positions, action affordance boundaries,
latent-action decoding into kept-share bids, `accept/reject/counter` labels,
initial resources, public knowledge, payoff transitions, raw outcome
resolution, and outcome utility feature names. Future environments should
provide a different adapter instead of editing `CognitiveAgent`.

### `action_model.py`

`CandidateInterventionGenerator` is the current action-formation module. It
combines actor situation, counterpart situation, directed relation edge, and
context through a neural network plus learnable action slots to produce latent
action vectors. The adapter grounds each vector into an executable action.

### `signal_model.py`

`SignalGenerator` creates an `OutgoingSignal` vector from physical action
encoding, responder intent, and actor situation. The vector has no fixed
semantic labels. It influences `BranchPolicy` through a learned linear head, so
meaning is acquired only through prediction/value/policy gradients.

Path dependence only uses recent response history. When root probabilities are
adjusted, descendant joint probabilities are scaled by the same factor. Preserve
that behavior when editing probability logic.

### `situation.py`

`HistorySummarizer` encodes `[signal || response_onehot || z_j]` and updates
history by EMA:

```text
h_j <- gamma * h_j + (1 - gamma) * tanh(W_enc(...))
```

`omega[0]` stores physical payoff/resource. The remaining resource coordinates
are initialized as small noise. `update_resource` exists as a helper; the demo
does not automatically commit payoff deltas into `omega`.

### `agent.py`

`CognitiveAgent` is the integration surface. It owns:

- `self.G`
- `self.interp`
- `self.bayes`
- `self.tolerance`
- `self.rules`
- `self.roles`
- `self.history`
- `self.tree`
- `self.action_gen`
- `self.signal_gen`
- `self.utility`
- `self.outcome_utility`
- `self.decision`
- mutable buffers: `self.h`, `self.omega`, `self.K`

When adding model components, prefer registering them here as modules or
buffers so optimizer behavior and device movement stay predictable.

`generate_candidate_interventions(...)` creates latent actions and adapter-
decoded interventions. `interpret_and_plan(action, ...)` generates an outgoing
signal and simulates the future conditioned on that action/signal pair.
`deliberate(...)` evaluates generated or supplied candidate interventions,
scores their predicted futures, and returns the selected action plus signal.
The public loop is `act(...) -> observe(response) -> learn(outcome)`.

### `decision.py`

`CandidateIntervention` represents an adapter-decoded action, optional latent
action vector, and outgoing communicative signal. `PredictedFuture` keeps each
`T(action, signal)` separate.
`FuturePositionEvaluator` is a learnable utility interface over planner
metrics. `DecisionEngine` turns predicted futures into candidate scores,
soft action probabilities, expected utility, and a selected action.

### `experience.py`

`Outcome` is the raw observed result after acting. `OutcomeUtilityEvaluator`
turns adapter-provided outcome features into realized utility. `LearningSignal`
keeps response prediction loss, value loss, and policy loss separate so outcome
is not treated as reward by accident. `ExperienceStep` is the
acted-and-observed record used by `CognitiveAgent.commit_experience`.

### `evaluation.py`

`AblationSpec` switches off pieces of the internal model during deliberation.
`NO_UNDERSTANDING` removes intent, outgoing signal, and actor situation.
`UsefulnessReport` compares full versus ablated decisions. The key evidence is
whether expected utility, candidate scores, or action probabilities change when
understanding is removed.

### `demo.py`

The demo is currently the main executable verification. It:

- seeds torch
- creates `CognitiveAgent`
- lets the agent choose bids and outgoing signals through `deliberate`
- resolves a raw `Outcome` and builds an `ExperienceStep`
- prints metrics and gradient norms
- asserts nonzero gradients for `G`, `InterpretationEngine`, and
  `FutureTreeGen.policy`, `CandidateInterventionGenerator`,
  `SignalGenerator`, and `FuturePositionEvaluator`

## How To Verify Changes

Use:

```powershell
python -B -m unittest discover -s tests -p "test_*.py"
python -B demo.py
python -B verify_architecture.py
```

Expected high-level result:

```text
OK: unittest contract and closed-loop regression tests pass.
OK: agent generated actions/signals and gradients flowed through G, I, T, A, S, U.
OK: ablating understanding changes expected position and policy scores.
```

Plain `python demo.py` should also work, but `python -B` avoids bytecode writes.
In this workspace, direct `py_compile` previously failed because Windows denied
renaming a file under `__pycache__`; that was a cache/write issue, not a source
syntax issue.

## Common Change Patterns

- **Adding a new context feature**: update `config.N_CONTEXT`, `config.M`,
  `config.INPUT_DIM`, `build_context`, and any tests/demo assumptions.
- **Adding a new response type**: update the adapter's `response_labels`,
  transition semantics, and outcome feature logic. `HistorySummarizer` and
  `BranchPolicy` derive their sizes from the adapter.
- **Changing action formation**: edit `action_model.py` for latent generation
  and the adapter's `action_affordance` / `decode_action` for grounding.
- **Changing dimensions**: update `config.py` first and keep `sanity_check`
  strict. Then update all dependent layer widths.
- **Changing tree value**: edit `FutureTreeGen._quality` or `evaluate`, while
  preserving differentiability through probabilities.
- **Adding training losses**: add them around the existing demo loss shape.
  Verify gradient norms still reach `G`, `I`, `T.policy`, and `action_gen`.

## Known Caveats

- Some comments/docstrings may show mojibake around Chinese labels in certain
  terminals. Treat that as display/encoding noise unless the user explicitly
  asks to clean it up.
- There is no dedicated test suite yet. `demo.py` is the regression smoke test.
- Imports are top-level local imports, not package-relative imports.
- The repository is intentionally small. Avoid adding framework scaffolding
  unless it solves an actual project need.

## Current Baseline

As of the most recent agent pass, all Python files were read and
`python -B demo.py` completed successfully. The demo reported nonzero gradient
norms for the core structures plus the utility evaluator and ended with:

```text
OK: agent generated actions/signals and gradients flowed through G, I, T, A, S, U.
```

The active architecture goal is broader than this baseline. See
`docs/first-principles-ai-architecture.md` for the target design. Current
implemented slices: the agent can generate latent actions, ground them through
the adapter, generate learned outgoing signals, simulate candidate
action/signal interventions, score predicted futures, choose its own
intervention, resolve a raw observed outcome, evaluate realized utility, expose
`act/observe/learn`, and separate response-prediction, value, and policy
losses. Toy-game-specific roles/affordances/responses/outcome features now live
in `UltimatumGameAdapter` instead of the core loop. The broader goal remains
active: richer experience-driven value/preference learning and stronger
multi-scenario tests for understanding-as-action improvement still need to be
added.
