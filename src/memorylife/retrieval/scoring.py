"""Retrieval scoring (the "sim(q,e) + alpha*i^ + beta*utility" box):
combines raw semantic similarity with the Importance heuristic and the
Future-utility head's prediction, so retrieval doesn't just surface the
most semantically similar memory -- it also prefers memories predicted to
matter and to actually be useful going forward. Weights are configurable
(configs/retrieval/*.yaml), not hardcoded, so sim_only.yaml (baseline,
alpha=beta=0) vs sim_importance_utility.yaml (ours) is a real ablation.
"""
from dataclasses import dataclass

from ..memory.memory_object import MemoryObject


@dataclass
class ScoringWeights:
    alpha: float = 0.3  # importance weight
    beta: float = 0.3  # utility weight


def combined_score(similarity: float, obj: MemoryObject, weights: ScoringWeights = ScoringWeights()) -> float:
    return similarity + weights.alpha * obj.importance + weights.beta * obj.utility_prob
