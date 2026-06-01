"""Original three-structure architecture (ultimatum game), clean re-implementation.

    ① G  人际关系图    relation_graph.RelationGraph
    ② T  未来结果树    future_tree.FutureTreeGen
    ③ I  解释机制      interpretation.InterpretationEngine

All e-commerce / LLM machinery is intentionally absent. See README.md.
"""

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
from game_adapter import EntitySet, UltimatumGameAdapter
from future_tree import FutureTreeGen, BranchPolicy, Node
from signal_model import OutgoingSignal, SignalGenerator
from action_model import GeneratedInterventions, CandidateInterventionGenerator
from decision import (
    CandidateIntervention, PredictedFuture, DecisionResult,
    FuturePositionEvaluator, DecisionEngine,
)
from experience import (
    Outcome, RealizedUtility, LearningSignal, ExperienceStep,
    OutcomeFeatureEncoder, OutcomeUtilityEvaluator,
    resolve_ultimatum_outcome, build_learning_signal,
)
from evaluation import (
    AblationSpec, UsefulnessReport, NO_UNDERSTANDING, NO_SIGNAL,
    compare_decisions,
)
from agent import CognitiveAgent

__all__ = [
    "RelationGraph",
    "InterpretationEngine", "RuleInterpretation", "BayesianInverse",
    "ToleranceHead", "build_signal", "build_context", "dissonance_loss",
    "RoleEmbedding", "HistorySummarizer", "init_resource", "update_resource",
    "init_knowledge", "build_sigma",
    "UltimatumRule",
    "EntitySet", "UltimatumGameAdapter",
    "FutureTreeGen", "BranchPolicy", "Node",
    "OutgoingSignal", "SignalGenerator",
    "GeneratedInterventions", "CandidateInterventionGenerator",
    "CandidateIntervention", "PredictedFuture", "DecisionResult",
    "FuturePositionEvaluator", "DecisionEngine",
    "Outcome", "RealizedUtility", "LearningSignal", "ExperienceStep",
    "OutcomeFeatureEncoder", "OutcomeUtilityEvaluator",
    "resolve_ultimatum_outcome", "build_learning_signal",
    "AblationSpec", "UsefulnessReport", "NO_UNDERSTANDING", "NO_SIGNAL",
    "compare_decisions",
    "CognitiveAgent",
]
