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
from game_spec import (
    ActionAffordance, ControlSpec, EntitySpec, FeatureSpec, GameSpec,
    GroundedAction, ResponseSpec, RoleBinding, StateVarSpec, TransitionSpec,
)
from generic_adapter import GenericGameAdapter
from game_rule import UltimatumRule
from game_spec import EntitySet
from future_tree import (
    FutureTreeGen, BranchPolicy, Node, WorldResponseDistribution,
    WorldResponseModel, CounterfactualPlanner, FutureValueModel,
)
from signal_model import OutgoingSignal, SignalGenerator
from action_model import (
    ActionPolicyOutput, GeneratedInterventions,
    CandidateInterventionGenerator,
)
from belief import BeliefState
from runtime import (
    ActionEvent, CheckpointMetadata, Observation, ObservationSpec,
    RuntimeSchema, RuntimeSnapshot, SchemaCompatibilityReport,
    TerminalOutcome, TransitionResult, WorldResponse,
    check_schema_compatibility,
)
from trajectory import (
    LearningCoordinator, OutcomeTargetBuilder, ReturnBuilder, ReturnTarget,
    Trajectory, TrajectoryStep,
)
from decision import (
    CandidateIntervention, PredictedFuture, DecisionResult,
    FuturePositionEvaluator, DecisionEngine,
)
from experience import (
    Outcome, RealizedUtility, LearningSignal, ExperienceStep,
    OutcomeFeatureEncoder, OutcomeUtilityEvaluator,
    build_learning_signal,
)
from evaluation import (
    AblationSpec, UsefulnessReport, NO_UNDERSTANDING, NO_SIGNAL,
    ArchitectureGateReport, compare_decisions, runtime_gate_report,
)
from agent import CognitiveAgent

__all__ = [
    "RelationGraph",
    "InterpretationEngine", "RuleInterpretation", "BayesianInverse",
    "ToleranceHead", "build_signal", "build_context", "dissonance_loss",
    "RoleEmbedding", "HistorySummarizer", "init_resource", "update_resource",
    "init_knowledge", "build_sigma",
    "ActionAffordance", "ControlSpec", "EntitySpec", "FeatureSpec",
    "GameSpec", "GroundedAction", "ResponseSpec", "RoleBinding",
    "StateVarSpec", "TransitionSpec", "GenericGameAdapter",
    "UltimatumRule",
    "EntitySet",
    "FutureTreeGen", "BranchPolicy", "Node", "WorldResponseDistribution",
    "WorldResponseModel", "CounterfactualPlanner", "FutureValueModel",
    "OutgoingSignal", "SignalGenerator",
    "ActionPolicyOutput", "GeneratedInterventions",
    "CandidateInterventionGenerator",
    "BeliefState",
    "ActionEvent", "CheckpointMetadata", "Observation", "ObservationSpec",
    "RuntimeSchema", "RuntimeSnapshot", "SchemaCompatibilityReport",
    "TerminalOutcome", "TransitionResult", "WorldResponse",
    "check_schema_compatibility",
    "LearningCoordinator", "OutcomeTargetBuilder", "ReturnBuilder",
    "ReturnTarget", "Trajectory", "TrajectoryStep",
    "CandidateIntervention", "PredictedFuture", "DecisionResult",
    "FuturePositionEvaluator", "DecisionEngine",
    "Outcome", "RealizedUtility", "LearningSignal", "ExperienceStep",
    "OutcomeFeatureEncoder", "OutcomeUtilityEvaluator",
    "build_learning_signal",
    "AblationSpec", "UsefulnessReport", "ArchitectureGateReport",
    "NO_UNDERSTANDING", "NO_SIGNAL", "compare_decisions",
    "runtime_gate_report",
    "CognitiveAgent",
]
