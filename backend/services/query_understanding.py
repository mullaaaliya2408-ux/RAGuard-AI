"""
Query Understanding Agent (spec Agent, pre-retrieval).

Before we retrieve anything, we ask the LLM to analyze the query: what is
the user actually asking, what type of question is it, what entities/
keywords matter, and is the query itself too vague to retrieve well as-is
(e.g. "what is attendance?" vs "what is the minimum attendance percentage
required at this college?").

This runs once per query, before the retrieval loop starts.
"""

import logging

from models.schemas import QueryAnalysis
from services.llm_client import generate_json

logger = logging.getLogger("raguard.query_understanding")

SYSTEM_INSTRUCTION = """You are a query analysis component in a document Q&A system.
Given a user's question, analyze it and return ONLY a JSON object with these exact keys:
- intent: one short label describing what the user wants (e.g. "policy_lookup", "definition", "comparison", "procedure", "numeric_fact")
- question_type: one of "factual", "yes_no", "how_to", "comparison", "list", "other"
- entities: list of specific named things mentioned (proper nouns, specific terms)
- keywords: list of important content words, excluding stopwords
- document_references: list of any document names explicitly mentioned, else []
- is_ambiguous: true if the query is too vague/short to retrieve precisely (e.g. missing what document, what specific policy, what time period)
- rewritten_query: if is_ambiguous is true, provide a clearer, more specific version of the query. Otherwise null.

Return ONLY valid JSON, no other text."""


def analyze_query(query: str) -> QueryAnalysis:
    """Run the Query Understanding Agent on a raw user query."""
    prompt = f'User question: "{query}"'

    result = generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)

    if not result:
        # Fail safe: if the LLM call fails, fall back to a minimal analysis
        # rather than crashing the whole pipeline.
        logger.warning("Query understanding failed, falling back to defaults")
        return QueryAnalysis(
            original_query=query,
            intent="unknown",
            question_type="other",
        )

    return QueryAnalysis(
        original_query=query,
        intent=result.get("intent", "unknown"),
        question_type=result.get("question_type", "other"),
        entities=result.get("entities", []),
        keywords=result.get("keywords", []),
        document_references=result.get("document_references", []),
        is_ambiguous=result.get("is_ambiguous", False),
        rewritten_query=result.get("rewritten_query"),
    )