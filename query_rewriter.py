"""
Query Rewriter
===============
Rewrites user queries to improve retrieval when context is weak or irrelevant.
Uses multiple strategies: keyword expansion, decomposition, and synonym enrichment.
"""

import re
from typing import List


def rewrite_query(original_query: str, attempt: int = 1) -> str:
    """
    Rewrite a query to improve retrieval results.

    Strategy varies by attempt number:
    - Attempt 1: Expand with synonyms and related terms
    - Attempt 2: Decompose into sub-questions and simplify

    Parameters
    ----------
    original_query : the user's original question
    attempt : which rewrite attempt this is (1-based)

    Returns
    -------
    A rewritten query string
    """
    query = original_query.strip()

    if attempt == 1:
        return _expand_query(query)
    elif attempt >= 2:
        return _simplify_query(query)
    else:
        return query


def _expand_query(query: str) -> str:
    """
    Expand the query by adding context keywords.
    Strips question words, adds the core noun phrases back,
    and broadens the search surface.
    """
    # Remove common question starters
    cleaned = re.sub(
        r'^(what is|what are|how does|how do|how to|why does|why do|'
        r'explain|describe|define|tell me about|can you explain|'
        r'what does|who is|who are|when did|where is)\s+',
        '',
        query.lower(),
    )

    # Extract key terms (words > 3 chars, not stop words)
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
        'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
        'have', 'been', 'this', 'that', 'with', 'from', 'they',
        'will', 'each', 'make', 'like', 'does', 'into', 'them',
        'then', 'than', 'what', 'when', 'where', 'which', 'their',
        'about', 'would', 'there', 'could', 'other', 'more',
    }
    words = re.findall(r'\b\w+\b', cleaned)
    key_terms = [w for w in words if len(w) > 3 and w not in stop_words]

    if key_terms:
        # Build expanded query: original core + key terms repeated for emphasis
        expanded = f"{cleaned} {' '.join(key_terms)}"
        return expanded.strip()

    return query


def _simplify_query(query: str) -> str:
    """
    Simplify a complex query into its core information need.
    Removes filler, focuses on key concepts.
    """
    # Remove question marks and trailing punctuation
    cleaned = query.rstrip('?!.').strip()

    # Remove filler phrases
    filler_patterns = [
        r'\b(please|kindly|could you|can you|i want to know|'
        r'i need to understand|tell me|i am curious about|'
        r'i would like to know)\b',
    ]
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Collapse whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # If the result is too short, fall back to extracting key terms
    if len(cleaned.split()) <= 2:
        words = re.findall(r'\b\w{4,}\b', query.lower())
        if words:
            return ' '.join(words)

    return cleaned


def get_alternative_queries(original_query: str) -> List[str]:
    """
    Generate multiple alternative versions of a query.
    Useful for multi-query retrieval strategies.
    """
    alternatives = []

    # Strategy 1: Expanded version
    alternatives.append(_expand_query(original_query))

    # Strategy 2: Simplified version
    alternatives.append(_simplify_query(original_query))

    # Strategy 3: Key terms only
    words = re.findall(r'\b\w{4,}\b', original_query.lower())
    stop_words = {'what', 'when', 'where', 'which', 'their', 'about', 'would', 'there', 'could', 'other', 'more', 'does', 'have', 'been', 'this', 'that', 'with', 'from', 'they', 'will', 'each', 'make', 'like', 'into', 'them', 'then', 'than'}
    key_only = ' '.join(w for w in words if w not in stop_words)
    if key_only:
        alternatives.append(key_only)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in alternatives:
        q_normalized = q.strip().lower()
        if q_normalized and q_normalized not in seen:
            seen.add(q_normalized)
            unique.append(q.strip())

    return unique

