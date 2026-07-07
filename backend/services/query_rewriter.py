"""
Adaptive Query Rewriter (spec: rewrite when retrieval confidence is low).

When the Retrieval Quality Checker fails a query, this agent generates
alternative phrasings — using the Query Understanding Agent's analysis
(entities, keywords, intent) as context — and the Self-Correction Loop
retries retrieval with each until one clears the quality bar or attempts
run out.
"""

import logging

from models.schemas import QueryAnalysis, QualityAssessment
from services.llm_client import generate_json

logger = logging.getLogger("raguard.query_rewriter")

SYSTEM_INSTRUCTION = """You are a query rewriting component in a document Q&A system.
The user's original query did not retrieve strong evidence from the document index.
Given the original query and its analysis, generate 2-3 alternative phrasings that are
more likely to match specific document text — e.g. more specific terminology, alternate
synonyms, or a more literal keyword-focused phrasing.

Return ONLY a JSON object with key "rewrites": a list of 2-3 alternative query strings.
Do not repeat the original query."""


def rewrite_query(original_query: str, analysis: QueryAnalysis, quality: QualityAssessment) -> list[str]:
    """Generate alternative query phrasings to retry retrieval with."""
    prompt = f"""Original query: "{original_query}"
Detected intent: {analysis.intent}
Keywords: {analysis.keywords}
Entities: {analysis.entities}
Why retrieval was weak: {quality.reasons}
"""

    result = generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
    rewrites = result.get("rewrites", [])

    if not rewrites:
        # Fallback: at minimum, try the keyword-only version we already
        # computed cheaply in Milestone 4's query_expansion module.
        from services.query_expansion import generate_keyword_variant
        fallback = generate_keyword_variant(original_query)
        rewrites = [fallback] if fallback != original_query else []
        logger.warning("LLM rewrite failed, using keyword-only fallback")

    logger.info(f"Rewritten queries: {rewrites}")
    return rewrites