"""
Lightweight query expansion: generates a small set of query variants to
widen retrieval recall. This is heuristic-based for now — Milestone 5's
Query Understanding Agent adds LLM-driven rewriting on top of this when
retrieval confidence is low. Keeping this separate means the agent can
call this same function as one of its tools later.
"""

import re

# Common stopwords stripped when generating a "keywords-only" variant
STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "for", "to",
    "how", "do", "does", "can", "i", "my", "your", "please", "tell", "me",
}


def generate_keyword_variant(query: str) -> str:
    """Strip stopwords/question framing to produce a dense keyword query,
    e.g. "What is attendance?" -> "attendance"."""
    words = re.findall(r"\w+", query.lower())
    keywords = [w for w in words if w not in STOPWORDS]
    return " ".join(keywords) if keywords else query


def generate_query_variants(query: str) -> list[str]:
    """
    Produce the original query plus a keyword-focused variant. Duplicate
    variants are dropped so we don't waste retrieval calls.
    """
    variants = [query]
    keyword_variant = generate_keyword_variant(query)

    if keyword_variant and keyword_variant.lower() != query.lower():
        variants.append(keyword_variant)

    return variants