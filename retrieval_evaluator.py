"""
Retrieval Evaluator
====================
Evaluates the quality of retrieved chunks and decides whether the context
is strong enough to generate an answer or needs corrective action.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from config import (
    RELEVANCE_THRESHOLD,
    MIN_RELEVANT_CHUNKS,
    WEAK_CONTEXT_RATIO,
)


class ContextQuality(Enum):
    """Quality classification of retrieved context."""
    STRONG = "strong"       # Enough relevant chunks → generate answer
    WEAK = "weak"           # Some relevant chunks but not enough → rewrite query
    IRRELEVANT = "irrelevant"  # No relevant chunks at all → rewrite or decline


@dataclass
class EvaluationResult:
    """Result of evaluating retrieved chunks."""
    quality: ContextQuality
    relevant_chunks: List[Dict[str, Any]]
    irrelevant_chunks: List[Dict[str, Any]]
    total_retrieved: int
    num_relevant: int
    relevance_ratio: float
    avg_similarity: float
    explanation: str


def evaluate_retrieval(
    retrieved_chunks: List[Dict[str, Any]],
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    min_relevant: int = MIN_RELEVANT_CHUNKS,
    weak_ratio: float = WEAK_CONTEXT_RATIO,
) -> EvaluationResult:
    """
    Evaluate the quality of retrieved chunks based on similarity scores.

    Classification Logic:
    ─────────────────────
    1. STRONG: relevance_ratio >= weak_ratio AND num_relevant >= min_relevant
    2. WEAK:   some relevant chunks exist but ratio < weak_ratio OR count < min_relevant
    3. IRRELEVANT: no relevant chunks at all

    Parameters
    ----------
    retrieved_chunks : list of chunk dicts with 'similarity' field
    relevance_threshold : minimum similarity to consider a chunk relevant
    min_relevant : minimum number of relevant chunks for STRONG
    weak_ratio : minimum ratio of relevant/total for STRONG

    Returns
    -------
    EvaluationResult with quality classification and details
    """
    if not retrieved_chunks:
        return EvaluationResult(
            quality=ContextQuality.IRRELEVANT,
            relevant_chunks=[],
            irrelevant_chunks=[],
            total_retrieved=0,
            num_relevant=0,
            relevance_ratio=0.0,
            avg_similarity=0.0,
            explanation="No chunks were retrieved.",
        )

    # Separate relevant from irrelevant
    relevant = [c for c in retrieved_chunks if c["similarity"] >= relevance_threshold]
    irrelevant = [c for c in retrieved_chunks if c["similarity"] < relevance_threshold]

    total = len(retrieved_chunks)
    num_relevant = len(relevant)
    ratio = num_relevant / total if total > 0 else 0.0
    avg_sim = (
        sum(c["similarity"] for c in retrieved_chunks) / total if total > 0 else 0.0
    )

    # Classify
    if num_relevant == 0:
        quality = ContextQuality.IRRELEVANT
        explanation = (
            f"None of the {total} retrieved chunks met the relevance threshold "
            f"({relevance_threshold}). Average similarity: {avg_sim:.3f}. "
            f"The query may not be related to the indexed documents."
        )
    elif num_relevant >= min_relevant and ratio >= weak_ratio:
        quality = ContextQuality.STRONG
        explanation = (
            f"{num_relevant}/{total} chunks are relevant (ratio: {ratio:.2f}). "
            f"Average similarity: {avg_sim:.3f}. Context is strong enough to answer."
        )
    else:
        quality = ContextQuality.WEAK
        reasons = []
        if num_relevant < min_relevant:
            reasons.append(
                f"only {num_relevant} relevant chunk(s) (need ≥{min_relevant})"
            )
        if ratio < weak_ratio:
            reasons.append(
                f"relevance ratio {ratio:.2f} < threshold {weak_ratio}"
            )
        explanation = (
            f"Context is weak: {'; '.join(reasons)}. "
            f"Average similarity: {avg_sim:.3f}. "
            f"Query rewriting may improve results."
        )

    return EvaluationResult(
        quality=quality,
        relevant_chunks=relevant,
        irrelevant_chunks=irrelevant,
        total_retrieved=total,
        num_relevant=num_relevant,
        relevance_ratio=round(ratio, 3),
        avg_similarity=round(avg_sim, 4),
        explanation=explanation,
    )

