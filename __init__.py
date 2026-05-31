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
from future_tree import FutureTreeGen, BranchPolicy, Node
from agent import CognitiveAgent

__all__ = [
    "RelationGraph",
    "InterpretationEngine", "RuleInterpretation", "BayesianInverse",
    "ToleranceHead", "build_signal", "build_context", "dissonance_loss",
    "RoleEmbedding", "HistorySummarizer", "init_resource", "update_resource",
    "init_knowledge", "build_sigma",
    "UltimatumRule",
    "FutureTreeGen", "BranchPolicy", "Node",
    "CognitiveAgent",
]
