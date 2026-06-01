"""CognitiveAgent -- wires the three structures into one optimizable module.

    G  RelationGraph        (structure ①)
    I  InterpretationEngine (structure ③)  + r_j, BayesianInverse, ToleranceHead
    T  FutureTreeGen        (structure ②)

Plus the per-agent situation pieces (role embedding, history, resource, K).

Everything is an nn.Module, so a single optimizer over `self.parameters()`
trains G, I and T jointly via autograd -- the kept optimization. The agent
exposes one `interpret_and_plan` step that runs the full v5 dynamic loop:

    s = phi(a, S, K)
    Z = [ I_j(s, G[j,i,:], r_j, sigma_j) ]_j          # interpretation
    z_C = propagate(z_B, ...)                          # re-signalling B -> C
    T   = Reconstruct(Z)                               # future tree
    z*  = BayesianInverse(s, z_prior)                  # dissonance target
"""

import torch
import torch.nn as nn

import config
from relation_graph import RelationGraph
from interpretation import (
    InterpretationEngine, RuleInterpretation, BayesianInverse, ToleranceHead,
    build_signal, build_context, dissonance_loss,
)
from situation import (
    RoleEmbedding, HistorySummarizer, init_resource, update_resource,
    init_knowledge, build_sigma,
)
from game_rule import UltimatumRule
from game_adapter import UltimatumGameAdapter
from future_tree import FutureTreeGen
from signal_model import SignalGenerator
from action_model import CandidateInterventionGenerator
from decision import (
    CandidateIntervention, PredictedFuture,
    DecisionEngine, FuturePositionEvaluator,
)
from experience import (
    OutcomeUtilityEvaluator, ExperienceStep,
    build_learning_signal,
)
from evaluation import compare_decisions


class CognitiveAgent(nn.Module):
    def __init__(self, rule=None, adapter=None, n_agents=None):
        super().__init__()
        self.rule = (
            rule if rule is not None
            else adapter.rule if adapter is not None and hasattr(adapter, "rule")
            else UltimatumRule())
        self.adapter = (
            adapter if adapter is not None else UltimatumGameAdapter(self.rule))
        self.n = (
            int(n_agents) if n_agents is not None
            else self.adapter.entities.n_entities)

        # structure ①
        self.G = RelationGraph(n_agents=self.n)

        # structure ③ + heads
        self.interp = InterpretationEngine()
        self.bayes = BayesianInverse()
        self.tolerance = ToleranceHead()
        self.rules = nn.ModuleList([
            RuleInterpretation(init_r=self.rule.r_public) for _ in range(self.n)
        ])

        # situation
        self.roles = RoleEmbedding(n_roles=self.n)
        self.history = HistorySummarizer(
            response_labels=self.adapter.response_labels())

        # structure ②  (shares the tolerance head so the tree reads B's z too)
        self.tree = FutureTreeGen(
            self.rule, adapter=self.adapter, tolerance_head=self.tolerance)
        self.action_gen = CandidateInterventionGenerator()
        self.signal_gen = SignalGenerator()
        self.utility = FuturePositionEvaluator()
        self.outcome_utility = OutcomeUtilityEvaluator(adapter=self.adapter)
        self.decision = DecisionEngine()

        # mutable per-episode situation buffers (not parameters)
        self.h = [torch.zeros(config.H_DIM) for _ in range(self.n)]
        initial_resources = (
            self.adapter.initial_resource_map()
            if hasattr(self.adapter, "initial_resource_map")
            else {i: 0.0 for i in range(self.n)})
        self.omega = [init_resource(i, initial_resources)
                      for i in range(self.n)]
        self.K = (
            self.adapter.initial_knowledge()
            if hasattr(self.adapter, "initial_knowledge")
            else init_knowledge())
        self.observed_history = []
        self._last_decision = None
        self._last_experience = None

    # ----- situation assembly ---------------------------------------------

    def sigma_of(self, j):
        rho = self.roles(j)
        return build_sigma(rho, self.h[j], self.omega[j], self.K)

    def r_of(self, j):
        return self.rules[j].r_j

    # ----- one full interpretive step -------------------------------------

    def interpret_and_plan(self, bid, actor_i=None, context=None,
                           depth=config.DEPTH, history=None, ablation=None):
        """Run the interpretive loop for a single grounded action.

        Returns a dict with Z, the propagated z_C, the future tree, and its
        evaluation metrics. Everything is differentiable.
        """
        if actor_i is None:
            actor_i = self.adapter.focal_actor
        s = build_signal(bid, context)
        r_dict = {j: self.r_of(j) for j in range(self.n)}
        sigma_dict = {j: self.sigma_of(j) for j in range(self.n)}

        Z = self.interp.compute_Z(s, actor_i, self.G, r_dict, sigma_dict)
        if ablation is not None and ablation.zero_intent:
            Z = torch.zeros_like(Z)

        # B -> C re-signalling (B tells the observer about A's offer)
        source = self.adapter.counterpart_actor
        target = self.adapter.observer_actor
        if target is None:
            z_C = torch.zeros_like(Z[source])
        else:
            z_C = self.interp.propagate(
                Z[source], self.G, source, target,
                r_dict[target], sigma_dict[target])
        if ablation is not None and ablation.zero_intent:
            z_C = torch.zeros_like(z_C)

        actor_sigma = sigma_dict[actor_i]
        if ablation is not None and ablation.zero_actor_situation:
            actor_sigma = torch.zeros_like(actor_sigma)

        outgoing_signal = self.signal_gen(
            s, Z[self.adapter.counterpart_actor], actor_sigma)
        comm_signal = outgoing_signal.vector
        if ablation is not None and ablation.zero_signal:
            comm_signal = torch.zeros_like(comm_signal)

        root = self.tree.simulate_action(
            Z, bid, sigma_root=sigma_dict[actor_i], depth=depth,
            comm_signal=comm_signal)
        self.tree.apply_path_dep(root, history, lam=config.LAMBDA)
        metrics = self.tree.evaluate(root, role=actor_i)
        return {
            "s": s,
            "Z": Z,
            "z_C": z_C,
            "outgoing_signal": outgoing_signal,
            "tree": root,
            "metrics": metrics,
        }

    # ----- first-principles decision loop ---------------------------------

    def generate_candidate_interventions(self, actor_i=None, context=None,
                                         ablation=None):
        """Generate latent actions and decode them through the adapter."""
        if actor_i is None:
            actor_i = self.adapter.focal_actor
        sigma_dict = {j: self.sigma_of(j) for j in range(self.n)}
        actor_sigma = sigma_dict[actor_i]
        if ablation is not None and ablation.zero_actor_situation:
            actor_sigma = torch.zeros_like(actor_sigma)
        counterpart = self.adapter.counterpart_actor
        relation_edge = self.G.get_edge(actor_i, counterpart)
        generated = self.action_gen(
            actor_sigma=actor_sigma,
            counterpart_sigma=sigma_dict[counterpart],
            relation_edge=relation_edge,
            context=context,
            adapter=self.adapter,
        )
        return generated

    def deliberate(self, actor_i=None, context=None,
                   candidate_actions=None, depth=config.DEPTH, history=None,
                   ablation=None):
        """Choose an action by simulating and scoring candidate interventions.

        Each candidate action is encoded as a signal, interpreted into Z,
        simulated as T(action), scored by the utility interface, and selected by
        the decision engine. This is the first code path where the agent chooses
        its own bid instead of receiving one from the demo.
        """
        generated = None
        if candidate_actions is None:
            generated = self.generate_candidate_interventions(
                actor_i=actor_i, context=context, ablation=ablation)
            candidate_actions = generated.candidates

        futures = []
        for candidate in candidate_actions:
            if not isinstance(candidate, CandidateIntervention):
                candidate = CandidateIntervention(
                    action=candidate, metadata={"source": "provided"})
            out = self.interpret_and_plan(
                candidate.action, actor_i=actor_i, context=context,
                depth=depth, history=history, ablation=ablation)
            outgoing_signal = out["outgoing_signal"]
            grounded_candidate = CandidateIntervention(
                action=candidate.action,
                signal=outgoing_signal,
                latent_action=candidate.latent_action,
                metadata=candidate.metadata,
            )
            futures.append(PredictedFuture(
                candidate=grounded_candidate,
                outgoing_signal=outgoing_signal,
                tree=out["tree"],
                metrics=out["metrics"],
                signal_vec=out["s"],
                Z=out["Z"],
                z_C=out["z_C"],
            ))

        decision = self.decision(futures, self.utility)
        return {
            "decision": decision,
            "futures": futures,
            "generated_interventions": generated,
        }

    def act(self, actor_i=None, context=None, candidate_actions=None,
            depth=config.DEPTH, history=None):
        """Public action surface: deliberate and remember the pending choice."""
        if history is None:
            history = self.observed_history
        out = self.deliberate(
            actor_i=actor_i, context=context,
            candidate_actions=candidate_actions, depth=depth, history=history)
        self._last_decision = out["decision"]
        return out

    def probe_understanding_usefulness(self, context=None, candidate_actions=None,
                                       depth=config.DEPTH, history=None,
                                       ablation=None):
        """Compare full deliberation against an ablated internal model."""
        if ablation is None:
            from evaluation import NO_UNDERSTANDING
            ablation = NO_UNDERSTANDING
        if history is None:
            history = self.observed_history
        full = self.deliberate(
            context=context, candidate_actions=candidate_actions,
            depth=depth, history=history)["decision"]
        ablated = self.deliberate(
            context=context, candidate_actions=candidate_actions,
            depth=depth, history=history, ablation=ablation)["decision"]
        return compare_decisions(full, ablated, ablation)

    # ----- post-action experience -----------------------------------------

    def resolve_outcome(self, action, response, pie=1.0):
        return self.adapter.resolve_outcome(action, response, pie=pie)

    def observe(self, response, decision=None, pie=1.0):
        """Public observation surface: convert a response into raw outcome."""
        decision = decision if decision is not None else self._last_decision
        if decision is None:
            raise ValueError("observe requires a decision or a previous act call.")
        return self.resolve_outcome(decision.selected.action, response, pie=pie)

    def evaluate_outcome(self, outcome, role=None,
                         device=None, dtype=torch.float32):
        if role is None:
            role = self.adapter.focal_actor
        return self.outcome_utility(
            outcome, role=role, device=device, dtype=dtype)

    def build_experience(self, decision, outcome, role=None):
        """Separate raw outcome, realized utility, and learning losses."""
        if role is None:
            role = self.adapter.focal_actor
        device = decision.scores.device
        dtype = decision.scores.dtype
        realized = self.evaluate_outcome(
            outcome, role=role, device=device, dtype=dtype)
        learning = build_learning_signal(decision, outcome, realized)
        return ExperienceStep(
            decision=decision,
            outcome=outcome,
            realized_utility=realized,
            learning_signal=learning,
        )

    def commit_experience(self, experience):
        """Commit observed outcome into mutable episode state."""
        future = experience.decision.selected_future
        outcome = experience.outcome
        self.update_history(future.signal_vec, outcome.response, future.Z)
        focal = self.adapter.focal_actor
        counterpart = self.adapter.counterpart_actor
        self.omega[focal] = update_resource(
            self.omega[focal],
            outcome.payoff_for(focal, adapter=self.adapter)).detach()
        self.omega[counterpart] = update_resource(
            self.omega[counterpart],
            outcome.payoff_for(counterpart, adapter=self.adapter)).detach()
        self.observed_history.append({"response": outcome.response})
        return self.h, self.omega

    def learn(self, outcome, decision=None, role=None):
        """Public learning surface: build and commit an experience step."""
        decision = decision if decision is not None else self._last_decision
        if decision is None:
            raise ValueError("learn requires a decision or a previous act call.")
        experience = self.build_experience(decision, outcome, role=role)
        self.commit_experience(experience)
        self._last_experience = experience
        return experience

    # ----- mutable episode history ----------------------------------------

    def update_history(self, signal_vec, response, Z, gamma=config.GAMMA):
        """Commit the observed response into each agent's history summary.

        `h` is episode state rather than a parameter. We detach the new summary
        so the next turn starts from the observation without backpropagating
        through previous optimizer steps.
        """
        for j in range(self.n):
            z_j = Z[j].detach()
            s_j = torch.as_tensor(
                signal_vec, dtype=z_j.dtype, device=z_j.device).detach()
            h_j = self.h[j].to(device=z_j.device, dtype=z_j.dtype)
            self.h[j] = self.history.update_h(
                h_j, s_j, response, z_j, gamma=gamma).detach()
        return self.h

    # ----- cognitive-dissonance loss --------------------------------------

    def dissonance(self, s, Z, observed_actor=None):
        """L = sum_j KL(z_j || z_j*), z_j* = BayesianInverse(s, z_j).

        The "real action" s corrects each observer's prior intent; the gap is
        the structural error the doc calls cognitive dissonance (认知失调).
        """
        if observed_actor is None:
            observed_actor = self.adapter.counterpart_actor
        total = Z.new_zeros(())
        for j in range(self.n):
            if j == observed_actor:
                continue
            z_j = Z[j]
            if torch.count_nonzero(z_j) == 0:
                continue
            z_star = self.bayes(s, z_j)
            total = total + dissonance_loss(z_j, z_star)
        return total

    def rule_reg(self):
        return sum(self.rules[j].regularization_loss(self.rule.r_public)
                   for j in range(self.n))
