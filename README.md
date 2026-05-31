# Original Three-Structure Game Cognition Architecture

This repository implements a differentiable cognitive architecture for the
Ultimatum Game. The system models how agents interpret an observed action,
how relational asymmetry shapes that interpretation, and how possible future
outcomes are reconstructed from the resulting intent representation.

The implementation is intentionally compact: every major component is a
PyTorch module, and the full loop can be optimized end to end with a single
optimizer.

## Core Idea

The architecture is built around three coupled structures:

| Structure | Module | Role |
|---|---|---|
| `G` Relation Graph | [relation_graph.py](relation_graph.py) | Stores directed, asymmetric relation vectors between agents. |
| `I` Interpretation Engine | [interpretation.py](interpretation.py) | Converts actions, relations, rule stance, and situation context into intent vectors. |
| `T` Future Outcome Tree | [future_tree.py](future_tree.py) | Reconstructs possible future branches from the interpreted intent matrix. |

The central representation is the intent matrix `Z in R^(n x d)`. Each row
`z_j` is agent `j`'s interpretation of another agent's action, conditioned on
its own relational edge, role, history, resources, public knowledge, and rule
stance.

## Architecture

### 1. Relation Graph `G`

`G` is a directed tensor of relation vectors:

```text
G[i, j, :] = agent i's relational view of agent j
G[i, j, :] != G[j, i, :]
```

This asymmetry is treated as a first-class computational resource. The graph is
stored as a single `nn.Parameter`, and edge values are clamped only at read time,
so gradient flow remains intact.

### 2. Interpretation Engine `I`

For each observing agent `j`, the interpretation engine computes:

```text
z_j = tanh(W_z [s || G[j, i, :] || r_j || sigma_j])
```

where:

- `s` is the encoded action signal.
- `G[j, i, :]` is observer `j`'s relation edge toward actor `i`.
- `r_j` is the observer's learnable rule interpretation.
- `sigma_j` is the observer's situation vector.

The situation vector is assembled as:

```text
sigma_j = [role_embedding || history_summary || resource_state || public_knowledge]
```

### 3. Future Outcome Tree `T`

The future tree is generated from the current intent matrix `Z`. It expands
candidate offers and possible responses:

```text
responses = accept / reject / counter
```

Branch probabilities are learned from the full responder intent vector `z_B`,
not from fixed semantic slices. The tree evaluates:

- `optionality`: normalized entropy over remaining leaf probability mass.
- `risk_floor`: minimum reachable leaf quality.
- `path_quality`: expected leaf quality.

Path dependence is applied before evaluation:

```text
P(branch | H) proportional to P(branch) * exp(lambda * consistency(branch, H))
```

When a root branch is reweighted, all descendant joint probabilities are scaled
accordingly, preserving coherent probability mass through counter branches.

## Innovations

### Differentiable Relational Asymmetry

The directed relation graph is not a static feature table. It is part of the
trainable computation graph. Gradients from interpretation loss and future-tree
value flow back into `G`, allowing relational structure to adapt through the
same optimizer as the rest of the model.

### Intent Without Fixed Semantic Slicing

The intent vector `z` is treated as a learned representation. Downstream modules
consume the full vector instead of assigning hard-coded meanings to ranges such
as `z[0:4]` or `z[4:12]`. This keeps the representation flexible and avoids
baking undocumented semantics into the architecture.

### Learned Branching Over Future Outcomes

The future tree is generated dynamically from interpretation state. Its branch
policy learns `P(response | z_B, offer)` from the responder's full intent vector,
the offer, and a learned tolerance head. The resulting tree remains
differentiable, so expected future value can train interpretation and relation
parameters.

### Action-Based Cognitive Dissonance

The model defines cognitive dissonance as a divergence between the current
intent representation and an action-conditioned inverse estimate:

```text
L = sum_j KL(z_j || z_j*)
z_j* = BayesianInverse(s, z_j)
```

The correction target is derived from the actual action signal, keeping the loss
grounded in observed behavior.

### Stateful Situation Encoding

Each agent carries a mutable history summary `h_j`, updated after observed
responses. This lets future interpretations depend on prior interaction
patterns while keeping the trainable model compact.

## End-to-End Flow

1. A proposer action is encoded as signal `s`.
2. Each observer interprets the action through its own relation edge and
   situation state.
3. The interpretation engine produces the intent matrix `Z`.
4. The responder's intent drives a generated future outcome tree.
5. Path dependence adjusts branch probabilities from recent history.
6. Tree metrics produce differentiable value signals.
7. Cognitive dissonance and rule regularization are combined with tree value.
8. A single optimizer updates the relation graph, interpretation engine, rule
   stances, tolerance head, and branch policy.

## File Map

| File | Purpose |
|---|---|
| [config.py](config.py) | Dimensions, game constants, and training hyperparameters. |
| [relation_graph.py](relation_graph.py) | Directed trainable relation graph `G`. |
| [interpretation.py](interpretation.py) | Signal construction, interpretation engine, inverse model, tolerance head. |
| [future_tree.py](future_tree.py) | Differentiable future-tree generation, path dependence, and evaluation. |
| [situation.py](situation.py) | Role embeddings, history summaries, resources, and public knowledge. |
| [game_rule.py](game_rule.py) | Ultimatum Game payoff and legality rules. |
| [agent.py](agent.py) | Composition layer that wires all structures into one trainable module. |
| [demo.py](demo.py) | End-to-end smoke test and gradient-flow verification. |

## Run

```bash
cd E:\game_algorithm_ori
python demo.py
```

The demo performs a short optimization loop and verifies that gradients flow
through all three core structures:

```text
G  RelationGraph
I  InterpretationEngine
T  FutureTreeGen.policy
```

## Dimensional Reference

```text
N_AGENTS = 3   A: proposer, B: responder, C: observer
K = 32         relation-edge vector dimension
D = 32         intent vector dimension
P = 16         rule-interpretation dimension

sigma = role(8) + history(16) + resource(8) + public_knowledge(8) = 40
s     = [bid, 1 - bid, fairness_deviation] + context(5) = 8

INPUT_DIM = s(8) + edge(32) + r_j(16) + sigma(40) = 96
```

## Minimal Example

```python
import torch

import config
from agent import CognitiveAgent
from interpretation import build_context

torch.manual_seed(config.SEED)

agent = CognitiveAgent()
context = build_context(turn_idx=0, session_len=8)
out = agent.interpret_and_plan(0.6, context=context, history=[])

print(out["Z"].shape)
print(out["metrics"])
```
