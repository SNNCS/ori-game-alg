"""Structure ②  未来结果树  T = (N, B, P).

Spec (v5, section 三):
    T = (N, B, P)
    N = state nodes (each a situation vector in R^d)
    B = branches (action, response, prob, new-node)
    P = branch probabilities, *driven by the interpretation matrix Z*
    T_{t+1} = Reconstruct(Z_t, G_t, H)        -- the tree is generated, not fixed

The v5 reference computed branch probability from a hard-coded slice:
    dignity = z_B[4:12].mean()
    P(reject) = dignity*0.8 + (action-0.5)*0.4   ...

We re-implement T and remove that slicing (user directive: z is not sliced).
Branch probabilities now come from a small *learnable* policy over the whole
responder intent vector z_B plus the offer (un)fairness, and (optionally) the
responder's tolerance head. Probabilities stay as torch tensors so gradients
flow from the tree's value back into the policy, the interpretation engine, and
the relation graph -- this is the "kept optimization" for T.

Metrics (v5):
    optionality   -- here a differentiable normalized entropy of leaf mass
    risk_floor    -- min leaf quality
    path_quality  -- expected leaf quality
    apply_path_dep-- P(b|H) ∝ P(b) * exp(lambda * consistency(b, H))
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from generic_adapter import GenericGameAdapter
from runtime import ActionEvent, RuntimeSnapshot, WorldResponse
from specs.ultimatum import build_ultimatum_spec


# ---------------------------------------------------------------------------
# Branch policy:  P(response | z_B, action)   -- learnable, no z slicing
# ---------------------------------------------------------------------------

class BranchPolicy(nn.Module):
    """Maps counterpart intent + intervention context to response probabilities.

    Replaces the v5 hard-coded `dignity = z_B[4:12].mean()` rule. The whole
    z_B is consumed by W_resp; the scalar (action-0.5) fairness gap is the
    only structural prior we keep (an unfairer grab should push toward reject),
    and even its effect is learned through w_action.
    """

    def __init__(self, d=config.D, n_responses=3, action_feature_dim=3,
                 signal_dim=config.SIGNAL_DIM, use_tolerance=True):
        super().__init__()
        self.use_tolerance = use_tolerance
        self.n_responses = int(n_responses)
        self.action_feature_dim = int(action_feature_dim)
        self.W_resp = nn.Linear(d, self.n_responses, bias=True)
        self.w_action = nn.Linear(self.action_feature_dim, self.n_responses,
                                  bias=False)
        self.w_signal = nn.Linear(signal_dim, self.n_responses, bias=False)
        if use_tolerance:
            self.w_tol = nn.Linear(1, self.n_responses, bias=False)

    def forward(self, z_B, action_features, tolerance=None, comm_signal=None):
        action_features = torch.as_tensor(
            action_features, device=z_B.device, dtype=z_B.dtype).reshape(-1)
        logits = self.W_resp(z_B) + self.w_action(action_features)
        if comm_signal is not None:
            logits = logits + self.w_signal(comm_signal.to(z_B.dtype))
        if self.use_tolerance and tolerance is not None:
            tol = tolerance.reshape(1).to(z_B.dtype)
            logits = logits + self.w_tol(tol)
        return F.softmax(logits, dim=-1)


@dataclass(frozen=True)
class WorldResponseDistribution:
    """Typed distribution for P(WorldResponse | belief, do(action))."""

    labels: tuple[str, ...]
    probabilities: torch.Tensor

    def responses(self):
        return tuple(WorldResponse(label=label) for label in self.labels)


class WorldResponseModel(nn.Module):
    """Names the world-response prediction responsibility explicitly."""

    def __init__(self, response_labels):
        super().__init__()
        self.response_labels = tuple(response_labels)

    def forward(self, policy, z_B, action_features, tolerance=None,
                comm_signal=None):
        probs = policy(
            z_B, action_features, tolerance=tolerance,
            comm_signal=comm_signal)
        return WorldResponseDistribution(
            labels=self.response_labels,
            probabilities=probs,
        )


class FutureValueModel(nn.Module):
    """Evaluate a simulated future through planner metrics."""

    def forward(self, tree_gen, root, role=None, utility_model=None):
        metrics = tree_gen.evaluate(root, role=role)
        value = (
            utility_model(metrics) if utility_model is not None
            else metrics["path_quality"])
        return value, metrics


class CounterfactualPlanner(nn.Module):
    """Simulate a candidate intervention from the current runtime snapshot."""

    def __init__(self, tree_gen):
        super().__init__()
        self.tree_gen = tree_gen

    def simulate(self, belief_state, action_event, snapshot=None,
                 depth=config.DEPTH, comm_signal=None):
        if belief_state.Z is None:
            raise ValueError(
                "Counterfactual planning requires action-conditioned belief.")
        action = (
            action_event.action if isinstance(action_event, ActionEvent)
            else action_event)
        responder = self.tree_gen.adapter.response_actor(
            snapshot=snapshot,
            action_event=action_event if isinstance(action_event, ActionEvent) else None,
        )
        return self.tree_gen._simulate_action_tree(
            belief_state.Z, action, depth=depth, comm_signal=comm_signal,
            snapshot=snapshot, responder=responder)


# ---------------------------------------------------------------------------
# Tree nodes
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("state", "action", "response", "prob", "children")

    def __init__(self, state, action=None, response=None, prob=None):
        self.state = state
        self.action = action
        self.response = response
        self.prob = prob              # torch scalar (keeps the graph alive)
        self.children = []


# ---------------------------------------------------------------------------
# FutureTreeGen
# ---------------------------------------------------------------------------

class FutureTreeGen(nn.Module):
    def __init__(self, rule=None, adapter=None, d=config.D, tolerance_head=None,
                 signal_dim=config.SIGNAL_DIM):
        super().__init__()
        self.adapter = (
            adapter if adapter is not None
            else GenericGameAdapter(build_ultimatum_spec()))
        self.rule = rule if rule is not None else self.adapter.rule
        self.d = d
        self.tolerance_head = tolerance_head      # optional ToleranceHead
        self.action_feature_dim = int(getattr(
            self.adapter, "branch_action_feature_dim", 3))
        self.policy = BranchPolicy(
            d=d, n_responses=len(self.adapter.response_labels()),
            action_feature_dim=self.action_feature_dim,
            signal_dim=signal_dim,
            use_tolerance=tolerance_head is not None)
        self.world_response_model = WorldResponseModel(
            self.adapter.response_labels())
        self.future_value_model = FutureValueModel()
        self.counterfactual_planner = CounterfactualPlanner(self)

    @property
    def RESPONSES(self):
        return self.adapter.response_labels()

    # ----- branch probabilities (driven by Z) ------------------------------

    def _branch_probs(self, z_B, action, comm_signal=None):
        tol = None
        if self.tolerance_head is not None:
            tol = self.tolerance_head(z_B)
        action_features = self.adapter.branch_action_features(
            action, device=z_B.device, dtype=z_B.dtype)
        return self.world_response_model(
            self.policy, z_B, action_features, tolerance=tol,
            comm_signal=comm_signal).probabilities

    def predict_world_responses(self, z_B, action, comm_signal=None):
        tol = None
        if self.tolerance_head is not None:
            tol = self.tolerance_head(z_B)
        action_features = self.adapter.branch_action_features(
            action, device=z_B.device, dtype=z_B.dtype)
        return self.world_response_model(
            self.policy, z_B, action_features, tolerance=tol,
            comm_signal=comm_signal)

    # ----- transition ------------------------------------------------------

    def transition(self, state, action, response):
        return self.adapter.transition(state, action, response)

    # ----- intent drift across depth --------------------------------------

    def update_Z(self, Z, response, responder=None):
        """Small whole-row perturbation of the responder's intent across a
        continue branch (v5 update_Z). No slicing -- the entire z_B row drifts.
        """
        Z2 = Z.clone()
        shift = self.adapter.intent_shift_for_response(response)
        row = (
            self.adapter.counterpart_actor
            if responder is None else int(responder))
        Z2[row, :] = torch.tanh(Z2[row, :] + shift)
        return Z2

    # ----- tree construction ----------------------------------------------

    def _initial_state(self, sigma_root=None, snapshot=None):
        if snapshot is not None:
            if not isinstance(snapshot, RuntimeSnapshot):
                snapshot = RuntimeSnapshot(state=snapshot)
            if hasattr(self.adapter, "_state_for_ctx"):
                state = self.adapter._state_for_ctx(snapshot.state)
            else:
                state = dict(snapshot.state)
                state["payoffs"] = dict(state.get("payoffs", {}))
            state["sigma"] = sigma_root
            return state
        return self.adapter.initial_tree_state(sigma_root)

    def _simulate_action_tree(self, Z, action, sigma_root=None,
                              depth=config.DEPTH, comm_signal=None,
                              snapshot=None, responder=None):
        """Build T(action): a future tree conditioned on one intervention.

        This is the planner primitive needed by a decision engine: it answers
        "if I do this action from this snapshot, what world responses and
        terminal/continuation states are possible?"
        """
        root_state = self._initial_state(sigma_root, snapshot=snapshot)
        if responder is None:
            responder = self.adapter.response_actor(snapshot=snapshot)
        return self._build_action(
            Z, root_state, action, depth, path_prob=None,
            comm_signal=comm_signal, responder=responder)

    def _build_action(self, Z, state, action, depth, path_prob,
                      comm_signal=None, responder=None):
        root = Node(state, prob=path_prob)
        responder = (
            self.adapter.counterpart_actor
            if responder is None else int(responder))
        z_B = Z[responder]
        probs = self._branch_probs(z_B, action, comm_signal=comm_signal)
        for ri, resp in enumerate(self.RESPONSES):
            p = probs[ri]
            joint = p if path_prob is None else path_prob * p
            new_state = self.transition(state, action, resp)
            child = Node(new_state, action=action, response=resp, prob=joint)
            root.children.append(child)
            if depth > 1 and self.adapter.is_continue_response(resp):
                next_action = self.adapter.continuation_action(
                    new_state, action, resp)
                if next_action is None:
                    continue
                new_Z = self.update_Z(Z, resp, responder=responder)
                sub = self._build_continue(
                    new_Z, new_state, next_action, depth - 1, joint,
                    comm_signal=comm_signal, responder=responder)
                child.children = sub.children
        return root

    def _build_continue(self, Z, state, action, depth, path_prob,
                        comm_signal=None, responder=None):
        root = Node(state, prob=path_prob)
        responder = (
            self.adapter.counterpart_actor
            if responder is None else int(responder))
        z_B = Z[responder]
        probs = self._branch_probs(z_B, action, comm_signal=comm_signal)
        for ri, resp in enumerate(self.RESPONSES):
            p = probs[ri]
            joint = path_prob * p
            new_state = self.transition(state, action, resp)
            child = Node(new_state, action=action, response=resp, prob=joint)
            root.children.append(child)
            if depth > 1 and self.adapter.is_continue_response(resp):
                next_action = self.adapter.continuation_action(
                    new_state, action, resp)
                if next_action is None:
                    continue
                new_Z = self.update_Z(Z, resp, responder=responder)
                sub = self._build_continue(
                    new_Z, new_state, next_action, depth - 1, joint,
                    comm_signal=comm_signal, responder=responder)
                child.children = sub.children
        return root

    # ----- evaluation ------------------------------------------------------

    def _collect_leaves(self, node, leaves):
        if not node.children:
            leaves.append(node)
            return
        for c in node.children:
            self._collect_leaves(c, leaves)

    def evaluate(self, root, role=None):
        """Returns a dict of torch scalars (differentiable w.r.t. the policy,
        interpretation engine, and relation graph through the leaf probs).
        """
        if role is None:
            role = self.adapter.focal_actor
        leaves = []
        self._collect_leaves(root, leaves)
        P = torch.stack([leaf.prob for leaf in leaves])
        P = P / (P.sum() + 1e-8)
        Q = torch.stack([
            torch.as_tensor(
                self.adapter.outcome_quality(leaf.state, role),
                dtype=P.dtype,
                device=P.device,
            ).reshape(())
            for leaf in leaves
        ])
        H = -(P * torch.log(P + 1e-12)).sum()
        H_max = torch.log(torch.tensor(float(max(len(P), 1)),
                                       dtype=P.dtype, device=P.device)) + 1e-12
        return {
            "optionality":  H / H_max,                 # normalized entropy in [0,1]
            "risk_floor":   Q.min(),
            "path_quality": (P * Q).sum(),
        }

    # ----- path dependence -------------------------------------------------

    @staticmethod
    def _scale_descendant_probs(node, scale):
        for child in node.children:
            child.prob = child.prob * scale
            FutureTreeGen._scale_descendant_probs(child, scale)

    def apply_path_dep(self, root, history, lam=config.LAMBDA):
        """P(b|H) ∝ P(b) * exp(lambda * consistency(b, H)) over root branches.

        Node probabilities are stored as joint path probabilities. When a root
        branch is reweighted, every descendant path under that branch must be
        scaled by the same factor or counter subtrees keep the stale mass.
        """
        if not history:
            return root
        recent = history[-10:]
        fallback = self.RESPONSES[-1]
        outs = [h.get("response", fallback) for h in recent]
        n = max(len(outs), 1)
        rate = {r: outs.count(r) / n for r in self.RESPONSES}

        def consistency(resp):
            return rate.get(resp, 0.0)

        new_probs = []
        for c in root.children:
            factor = torch.exp(torch.tensor(lam * consistency(c.response),
                                            dtype=c.prob.dtype, device=c.prob.device))
            new_probs.append(c.prob * factor)
        total = torch.stack(new_probs).sum() + 1e-8
        for c, np_ in zip(root.children, new_probs):
            old_prob = c.prob
            adjusted = np_ / total
            scale = adjusted / (old_prob + 1e-8)
            c.prob = adjusted
            self._scale_descendant_probs(c, scale)
        return root
