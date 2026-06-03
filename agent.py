"""CognitiveAgent -- wires the three structures into one optimizable module.

    G  RelationGraph        (structure ①)
    I  InterpretationEngine (structure ③)  + r_j, BayesianInverse, ToleranceHead
    T  FutureTreeGen        (structure ②)

Plus the per-agent situation pieces (role embedding, history, resource, K).

Everything is an nn.Module, so a single optimizer over `self.parameters()`
trains G, I and T jointly via autograd -- the kept optimization. The agent
plans from a runtime snapshot through this dynamic loop:

    s = phi(a, S, K)
    Z = [ I_j(s, G[j,i,:], r_j, sigma_j) ]_j          # interpretation
    z_C = propagate(z_B, ...)                          # re-signalling B -> C
    T   = Reconstruct(Z)                               # future tree
    z*  = BayesianInverse(s, z_prior)                  # dissonance target
"""

from dataclasses import replace

import torch
import torch.nn as nn

import config
from belief import BeliefState
from relation_graph import RelationGraph
from interpretation import (
    InterpretationEngine, RuleInterpretation, BayesianInverse, ToleranceHead,
    build_context, dissonance_loss,
)
from situation import (
    RoleEmbedding, HistorySummarizer, init_resource, update_resource,
    init_knowledge, build_sigma,
)
from generic_adapter import GenericGameAdapter
from future_tree import FutureTreeGen
from signal_model import SignalGenerator
from action_model import CandidateInterventionGenerator
from decision import (
    CandidateIntervention, PredictedFuture,
    DecisionEngine, FuturePositionEvaluator,
)
from experience import (
    OutcomeUtilityEvaluator, ExperienceStep,
)
from evaluation import compare_decisions
from runtime import ActionEvent, RuntimeSnapshot, WorldResponse
from specs.ultimatum import build_ultimatum_spec
from trajectory import LearningCoordinator, Trajectory, TrajectoryStep


class CognitiveAgent(nn.Module):
    def __init__(self, rule=None, adapter=None, n_agents=None):
        super().__init__()
        self.adapter = (
            adapter if adapter is not None
            else GenericGameAdapter(build_ultimatum_spec()))
        self.rule = rule if rule is not None else self.adapter.rule
        self.n = (
            int(n_agents) if n_agents is not None
            else self.adapter.entities.n_entities)
        self.action_signal_dim = int(getattr(
            self.adapter, "action_signal_dim", config.M))
        self.interp_input_dim = (
            self.action_signal_dim + config.K + config.P + config.SIGMA_DIM)

        # structure ①
        self.G = RelationGraph(n_agents=self.n)

        # structure ③ + heads
        self.interp = InterpretationEngine(
            input_dim=self.interp_input_dim,
            signal_dim=self.action_signal_dim,
        )
        self.bayes = BayesianInverse(m=self.action_signal_dim)
        self.tolerance = ToleranceHead()
        self.rules = nn.ModuleList([
            RuleInterpretation(init_r=self.rule.r_public) for _ in range(self.n)
        ])

        # situation
        self.roles = RoleEmbedding(n_roles=self.n)
        self.history = HistorySummarizer(
            response_labels=self.adapter.response_labels(),
            signal_dim=self.action_signal_dim)

        # structure ②  (shares the tolerance head so the tree reads B's z too)
        self.tree = FutureTreeGen(
            self.rule, adapter=self.adapter, tolerance_head=self.tolerance)
        self.action_gen = CandidateInterventionGenerator()
        self.signal_gen = SignalGenerator(
            input_dim=self.action_signal_dim + config.D + config.SIGMA_DIM)
        self.utility = FuturePositionEvaluator()
        self.outcome_utility = OutcomeUtilityEvaluator(adapter=self.adapter)
        self.decision = DecisionEngine()
        self.learning = LearningCoordinator()

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
        self._last_generated_interventions = None
        self._last_action_event = None
        self._last_experience = None

    # ----- situation assembly ---------------------------------------------

    def sigma_of(self, j):
        rho = self.roles(j)
        return build_sigma(rho, self.h[j], self.omega[j], self.K)

    def r_of(self, j):
        return self.rules[j].r_j

    # ----- runtime observation / belief -----------------------------------

    def runtime_snapshot(self, sigma_root=None):
        if hasattr(self.adapter, "initial_runtime_snapshot"):
            return self.adapter.initial_runtime_snapshot(sigma_root=sigma_root)
        return RuntimeSnapshot(state=self.adapter.initial_tree_state(sigma_root))

    def observe(self, snapshot, viewer=None):
        if viewer is None:
            viewer = self.adapter.focal_actor
        return self.adapter.encode_observation(snapshot, viewer=viewer)

    def observe_runtime(self, snapshot, viewer=None):
        return self.observe(snapshot, viewer=viewer)

    def _observation_embedding(self, observation):
        vector = observation.vector * observation.mask.to(
            device=observation.vector.device, dtype=observation.vector.dtype)
        vector = vector[:config.K_DIM]
        if vector.numel() < config.K_DIM:
            vector = torch.cat([
                vector,
                torch.zeros(
                    config.K_DIM - vector.numel(),
                    device=vector.device,
                    dtype=vector.dtype),
            ])
        return vector

    def update_belief(self, observation, actor_i=None):
        """Build an observation-derived BeliefState.

        This step does not infer intent for a hypothetical action. Intent lives
        in `_interpret_action`, where a concrete ActionEvent/candidate action is
        available.
        """
        if actor_i is None:
            actor_i = self.adapter.focal_actor
        r_dict = {j: self.r_of(j) for j in range(self.n)}
        sigma_dict = {j: self.sigma_of(j) for j in range(self.n)}
        return BeliefState(
            observation=observation,
            actor=actor_i,
            observation_embedding=self._observation_embedding(observation),
            sigma=sigma_dict,
            rules=r_dict,
            metadata={"adapter": self.adapter.spec.name},
        )

    def _interpret_action(self, belief, action, actor_i=None, context=None,
                          responder=None):
        """Return a belief state enriched with action-conditioned intent."""
        if actor_i is None:
            actor_i = belief.actor
        if responder is None:
            responder = self.adapter.response_actor(actor=actor_i)
        s = self.adapter.encode_action_signal(action, context)
        r_dict = belief.rules
        sigma_dict = belief.sigma
        Z = self.interp.compute_Z(s, actor_i, self.G, r_dict, sigma_dict)
        source = int(responder)
        target = self.adapter.observer_actor
        if target is None:
            z_C = torch.zeros_like(Z[source])
        else:
            z_C = self.interp.propagate(
                Z[source], self.G, source, target,
                r_dict[target], sigma_dict[target])
        return replace(
            belief,
            actor=actor_i,
            action_signal=s,
            Z=Z,
            z_C=z_C,
            metadata={**belief.metadata, "interpreted_action": True},
        )

    # ----- one full runtime planning step ---------------------------------

    def _candidate_future(self, candidate, snapshot, actor_i, context,
                          depth, history, ablation):
        observation = self.observe(snapshot, viewer=actor_i)
        belief = self.update_belief(observation, actor_i=actor_i)
        responder = self.adapter.response_actor(snapshot=snapshot, actor=actor_i)
        belief = self._interpret_action(
            belief, candidate.action, actor_i=actor_i, context=context,
            responder=responder)
        Z = belief.Z
        z_C = belief.z_C
        if Z is None or z_C is None or belief.action_signal is None:
            raise RuntimeError("Candidate future requires action-conditioned belief.")
        if ablation is not None and ablation.zero_intent:
            Z = torch.zeros_like(Z)
            z_C = torch.zeros_like(z_C)
            belief = replace(belief, Z=Z, z_C=z_C)

        actor_sigma = belief.sigma[actor_i]
        if ablation is not None and ablation.zero_actor_situation:
            actor_sigma = torch.zeros_like(actor_sigma)

        outgoing_signal = self.signal_gen(
            belief.action_signal, Z[responder],
            actor_sigma)
        comm_signal = outgoing_signal.vector
        if ablation is not None and ablation.zero_signal:
            comm_signal = torch.zeros_like(comm_signal)

        action_event = self.adapter.ground_action(
            candidate.action, snapshot=snapshot, actor=actor_i)
        root = self.tree.counterfactual_planner.simulate(
            belief, action_event, snapshot=snapshot, depth=depth,
            comm_signal=comm_signal)
        self.tree.apply_path_dep(root, history, lam=config.LAMBDA)
        metrics = self.tree.evaluate(root, role=actor_i)
        grounded_candidate = CandidateIntervention(
            action=candidate.action,
            signal=outgoing_signal,
            latent_action=candidate.latent_action,
            metadata=candidate.metadata,
        )
        return PredictedFuture(
            candidate=grounded_candidate,
            outgoing_signal=outgoing_signal,
            tree=root,
            metrics=metrics,
            signal_vec=belief.action_signal,
            Z=Z,
            z_C=z_C,
            metadata={"actor": int(actor_i), "responder": int(responder)},
        )

    # ----- first-principles decision loop ---------------------------------

    def generate_candidate_interventions(self, actor_i=None, context=None,
                                         ablation=None, snapshot=None):
        """Generate latent actions and decode them through the adapter."""
        if snapshot is None:
            raise ValueError("generate_candidate_interventions requires a RuntimeSnapshot.")
        if actor_i is None:
            actor_i = (
                snapshot.current_actor
                if snapshot.current_actor is not None
                else self.adapter.focal_actor)
        sigma_dict = {j: self.sigma_of(j) for j in range(self.n)}
        actor_sigma = sigma_dict[actor_i]
        if ablation is not None and ablation.zero_actor_situation:
            actor_sigma = torch.zeros_like(actor_sigma)
        counterpart = self.adapter.counterpart_for_actor(actor_i)
        relation_edge = self.G.get_edge(actor_i, counterpart)
        affordance = self.adapter.action_affordance(snapshot, actor=actor_i)
        generated = self.action_gen(
            actor_sigma=actor_sigma,
            counterpart_sigma=sigma_dict[counterpart],
            relation_edge=relation_edge,
            context=context,
            adapter=self.adapter,
            n_candidates=affordance.n_candidates,
            legal_mask=(
                self.adapter.legal_action_mask(snapshot, actor=actor_i)
                if hasattr(self.adapter, "legal_action_mask") else None),
        )
        return generated

    def deliberate(self, actor_i=None, context=None,
                   candidate_actions=None, depth=config.DEPTH, history=None,
                   ablation=None, snapshot=None):
        """Choose an action by simulating and scoring candidate interventions.

        Each candidate action is encoded as a signal, interpreted into Z,
        simulated as T(action), scored by the utility interface, and selected by
        the decision engine. This is the first code path where the agent chooses
        its own bid instead of receiving one from the demo.
        """
        if snapshot is None:
            raise ValueError("deliberate requires a RuntimeSnapshot.")
        if history is None:
            history = self.observed_history
        if actor_i is None:
            actor_i = (
                snapshot.current_actor
                if snapshot.current_actor is not None
                else self.adapter.focal_actor)
        generated = None
        if candidate_actions is None:
            generated = self.generate_candidate_interventions(
                actor_i=actor_i, context=context, ablation=ablation,
                snapshot=snapshot)
            candidate_actions = generated.candidates

        futures = []
        for candidate in candidate_actions:
            if not isinstance(candidate, CandidateIntervention):
                candidate = CandidateIntervention(
                    action=candidate, metadata={"source": "provided"})
            futures.append(self._candidate_future(
                candidate, snapshot, actor_i, context, depth, history,
                ablation))

        legal_mask = None
        if generated is not None and generated.policy_output is not None:
            legal_mask = generated.policy_output.legal_mask
        elif hasattr(self.adapter, "legal_action_mask"):
            runtime_mask = self.adapter.legal_action_mask(
                snapshot, actor=actor_i)
            if runtime_mask is not None:
                runtime_mask = torch.as_tensor(runtime_mask, dtype=torch.bool)
                if runtime_mask.numel() == len(futures):
                    legal_mask = runtime_mask
        decision = self.decision(futures, self.utility, legal_mask=legal_mask)
        self._last_decision = decision
        self._last_generated_interventions = generated
        return decision

    def act(self, actor_i=None, context=None, candidate_actions=None,
            depth=config.DEPTH, history=None, snapshot=None):
        """Select and ground an ActionEvent for the active runtime snapshot."""
        decision = self.deliberate(
            actor_i=actor_i, context=context,
            candidate_actions=candidate_actions, depth=depth, history=history,
            snapshot=snapshot)
        log_prob = decision.log_action_probs[decision.selected_index]
        actor = (
            actor_i if actor_i is not None
            else snapshot.current_actor if snapshot.current_actor is not None
            else self.adapter.focal_actor)
        action_event = ActionEvent(
            actor=int(actor),
            action=decision.selected.action,
            label=decision.selected.action.display,
            controls=dict(decision.selected.action.controls),
            log_prob=log_prob,
            metadata={
                "decision_index": decision.selected_index,
                "legal": True,
                "policy_source": "decision",
            },
        )
        self._last_action_event = action_event
        return action_event

    def probe_understanding_usefulness(self, context=None, candidate_actions=None,
                                       depth=config.DEPTH, history=None,
                                       ablation=None, snapshot=None):
        """Compare full deliberation against an ablated internal model."""
        if snapshot is None:
            raise ValueError("probe_understanding_usefulness requires a RuntimeSnapshot.")
        if ablation is None:
            from evaluation import NO_UNDERSTANDING
            ablation = NO_UNDERSTANDING
        full = self.deliberate(
            context=context, candidate_actions=candidate_actions,
            depth=depth, history=history, snapshot=snapshot)
        ablated = self.deliberate(
            context=context, candidate_actions=candidate_actions,
            depth=depth, history=history, ablation=ablation,
            snapshot=snapshot)
        return compare_decisions(full, ablated, ablation)

    # ----- post-action experience -----------------------------------------

    def transition_runtime(self, snapshot, action_event, world_response):
        if not isinstance(world_response, WorldResponse):
            raise TypeError("transition_runtime requires a WorldResponse.")
        return self.adapter.transition_event(
            snapshot, action_event, world_response)

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
        learning = self.learning.build_single_step_signal(
            decision, outcome, realized)
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
        for role, payoff in dict(outcome.entity_payoffs or {}).items():
            role = int(role)
            if 0 <= role < self.n:
                self.omega[role] = update_resource(
                    self.omega[role], payoff).detach()
        self.observed_history.append({"response": outcome.response})
        return self.h, self.omega

    def build_trajectory_step(self, decision, action_event, transition,
                              role=None):
        outcome = self.adapter.outcome_from_transition(transition)
        experience = self.build_experience(decision, outcome, role=role)
        return TrajectoryStep(
            decision=decision,
            action_event=action_event,
            world_response=transition.world_response,
            transition=transition,
            outcome=outcome,
            realized_utility=experience.realized_utility,
        )

    def learn(self, trajectory, role=None):
        """Public learning surface over a typed trajectory."""
        if not isinstance(trajectory, Trajectory):
            raise TypeError("learn requires a Trajectory.")
        if not trajectory.steps:
            raise ValueError("learn requires at least one TrajectoryStep.")
        if role is None:
            role = self.adapter.focal_actor
        signal = self.learning.build_trajectory_signal(trajectory, role=role)
        step = trajectory.steps[-1]
        if step.outcome is None or step.realized_utility is None:
            raise ValueError("TrajectoryStep requires outcome and realized utility.")
        experience = ExperienceStep(
            decision=step.decision,
            outcome=step.outcome,
            realized_utility=step.realized_utility,
            learning_signal=signal,
        )
        self.commit_experience(experience)
        self._last_experience = experience
        return signal

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
            zero_rows = [
                j for j in range(self.n)
                if torch.count_nonzero(Z[j]) == 0
            ]
            observed_actor = zero_rows[0] if len(zero_rows) == 1 else None
        total = Z.new_zeros(())
        for j in range(self.n):
            if observed_actor is not None and j == observed_actor:
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
