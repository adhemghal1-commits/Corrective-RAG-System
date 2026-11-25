"""
Corrective RAG Pipeline
=========================
The main orchestrator that ties all components together:
  Query → Retrieve → Evaluate → (Rewrite?) → Generate Answer

This is the core engine of the Corrective RAG System.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from embedding_store import EmbeddingStore
from retrieval_evaluator import evaluate_retrieval, ContextQuality, EvaluationResult
from query_rewriter import rewrite_query
from answer_generator import generate_answer, llm_rewrite_query, format_context
from config import MAX_REWRITE_ATTEMPTS, TOP_K


@dataclass
class PipelineResult:
    """Complete result from a Corrective RAG pipeline run."""
    # Final answer
    answer: str
    answerable: bool
    sources: List[str]

    # Retrieval details
    original_query: str
    final_query: str
    was_rewritten: bool
    rewrite_attempts: int

    # Evaluation trace
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Context used for generation
    relevant_chunks: List[Dict[str, Any]] = field(default_factory=list)
    context_used: str = ""

    # Pipeline metadata
    status: str = "success"  # success, no_answer, error


class CorrectiveRAGPipeline:
    """
    End-to-end Corrective RAG pipeline.

    Flow:
    ─────
    1. User submits a question
    2. Retrieve top-K chunks from vector store
    3. Evaluate retrieval quality
    4. If STRONG → generate answer from verified context
    5. If WEAK/IRRELEVANT → rewrite query and go to step 2
    6. After max attempts, generate answer with best available context
       or report that the question cannot be answered
    """

    def __init__(self, store: Optional[EmbeddingStore] = None):
        self.store = store or EmbeddingStore()

    def run(self, question: str, top_k: int = TOP_K) -> PipelineResult:
        """
        Execute the full Corrective RAG pipeline for a question.

        Parameters
        ----------
        question : the user's question
        top_k : number of chunks to retrieve per attempt

        Returns
        -------
        PipelineResult with the answer and full trace
        """
        original_query = question.strip()
        current_query = original_query
        rewrite_attempts = 0
        evaluation_history = []
        best_relevant_chunks = []
        best_evaluation = None

        # ── Retrieval + Evaluation Loop ──────────
        for attempt in range(1 + MAX_REWRITE_ATTEMPTS):
            # Step 1: Retrieve
            retrieved_chunks = self.store.query(current_query, top_k=top_k)

            # Step 2: Evaluate
            evaluation = evaluate_retrieval(retrieved_chunks)

            eval_record = {
                "attempt": attempt + 1,
                "query": current_query,
                "quality": evaluation.quality.value,
                "num_relevant": evaluation.num_relevant,
                "total_retrieved": evaluation.total_retrieved,
                "relevance_ratio": evaluation.relevance_ratio,
                "avg_similarity": evaluation.avg_similarity,
                "explanation": evaluation.explanation,
            }
            evaluation_history.append(eval_record)

            # Keep track of the best set of relevant chunks across attempts
            if len(evaluation.relevant_chunks) > len(best_relevant_chunks):
                best_relevant_chunks = evaluation.relevant_chunks
                best_evaluation = evaluation

            # Step 3: Decide
            if evaluation.quality == ContextQuality.STRONG:
                # Context is good → proceed to generation
                break

            if attempt < MAX_REWRITE_ATTEMPTS:
                # Context is weak/irrelevant → rewrite and retry
                rewrite_attempts += 1
                current_query = rewrite_query(original_query, attempt=rewrite_attempts)

                # If the basic rewriter didn't change much, try LLM rewrite
                if current_query.lower().strip() == original_query.lower().strip():
                    try:
                        current_query = llm_rewrite_query(original_query)
                    except Exception:
                        pass  # Fall back to basic rewrite

        # ── Generation Phase ─────────────────────
        was_rewritten = rewrite_attempts > 0
        final_query = current_query

        # Use the best relevant chunks we found
        if best_relevant_chunks:
            gen_result = generate_answer(original_query, best_relevant_chunks)
        else:
            gen_result = generate_answer(original_query, [])

        # ── Build Result ─────────────────────────
        result = PipelineResult(
            answer=gen_result["answer"],
            answerable=gen_result["answerable"],
            sources=gen_result["sources"],
            original_query=original_query,
            final_query=final_query,
            was_rewritten=was_rewritten,
            rewrite_attempts=rewrite_attempts,
            evaluation_history=evaluation_history,
            relevant_chunks=best_relevant_chunks,
            context_used=gen_result.get("context_used", ""),
            status="success" if gen_result["answerable"] else "no_answer",
        )

        return result

    def ingest_documents(self, chunks: list) -> int:
        """
        Ingest processed document chunks into the vector store.
        Returns the number of chunks added.
        """
        return self.store.add_chunks(chunks)

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return self.store.get_collection_stats()

    def is_ready(self) -> bool:
        """Check if the pipeline has documents to search."""
        return self.store.has_documents()

    def clear(self):
        """Clear all documents from the vector store."""
        self.store.clear_collection()

