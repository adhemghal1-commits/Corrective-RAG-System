"""
Answer Generator
=================
Generates answers from verified context using an LLM.
Supports both local (HuggingFace) and OpenAI backends.
Includes source attribution and handles unanswerable questions.
"""

from typing import List, Dict, Any, Optional
from config import (
    USE_OPENAI,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LOCAL_LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)


# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────

ANSWER_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer the question using ONLY the information in the context below.
2. If the context does not contain enough information to answer, say "I cannot answer this question based on the available documents."
3. Be concise and accurate.
4. Reference which source document(s) the information comes from.
5. Do NOT make up information or use knowledge outside the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


REWRITE_PROMPT = """Rewrite the following question to be more specific and searchable.
The goal is to improve document retrieval.

Original question: {question}

Rewritten question:"""


# ──────────────────────────────────────────────
# LLM Backends
# ──────────────────────────────────────────────

_local_model = None
_local_tokenizer = None


def _get_local_model():
    """Lazy-load the local HuggingFace model."""
    global _local_model, _local_tokenizer
    if _local_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print(f"Loading local LLM: {LOCAL_LLM_MODEL} ...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_MODEL)
        print("Local LLM loaded.")
    return _local_model, _local_tokenizer


def _generate_local(prompt: str) -> str:
    """Generate text using the local HuggingFace model."""
    model, tokenizer = _get_local_model()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=max(TEMPERATURE, 0.01),
        do_sample=True,
        num_beams=2,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _generate_openai(prompt: str) -> str:
    """Generate text using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )
    return response.choices[0].message.content.strip()


def generate_text(prompt: str) -> str:
    """Generate text using the configured LLM backend."""
    if USE_OPENAI and OPENAI_API_KEY:
        return _generate_openai(prompt)
    else:
        return _generate_local(prompt)


# ──────────────────────────────────────────────
# Answer Generation
# ──────────────────────────────────────────────

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a context string with source labels."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        similarity = chunk.get("similarity", 0)
        context_parts.append(
            f"[Source {i}: {source} | Relevance: {similarity:.2f}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    question: str,
    relevant_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate an answer from verified context chunks.

    Returns a dict with:
    - answer: the generated answer text
    - sources: list of source documents used
    - context_used: the formatted context string
    - answerable: whether the question was answerable
    """
    if not relevant_chunks:
        return {
            "answer": (
                "I cannot answer this question based on the available documents. "
                "The retrieved context was not relevant enough to provide a reliable answer."
            ),
            "sources": [],
            "context_used": "",
            "answerable": False,
        }

    # Format context
    context = format_context(relevant_chunks)

    # Build prompt
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    # Generate answer
    answer = generate_text(prompt)

    # Extract unique sources
    sources = list(set(
        chunk.get("metadata", {}).get("source", "Unknown")
        for chunk in relevant_chunks
    ))

    # Check if the model indicated it cannot answer
    cannot_answer_phrases = [
        "cannot answer",
        "not enough information",
        "no information",
        "not mentioned",
        "does not contain",
        "unable to answer",
        "i don't have enough",
    ]
    answerable = not any(
        phrase in answer.lower() for phrase in cannot_answer_phrases
    )

    return {
        "answer": answer,
        "sources": sorted(sources),
        "context_used": context,
        "answerable": answerable,
    }


def llm_rewrite_query(question: str) -> str:
    """Use the LLM to rewrite a query for better retrieval."""
    prompt = REWRITE_PROMPT.format(question=question)
    rewritten = generate_text(prompt)
    return rewritten.strip() if rewritten.strip() else question

