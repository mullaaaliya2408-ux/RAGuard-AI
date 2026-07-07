"""
Contradiction Detector (spec Agent 5).

Compares retrieved chunks against each other to find cases where two
sources state conflicting facts about the same thing (e.g. Document A says
"minimum 75% attendance", Document B says "minimum 80% attendance").

This is a real risk in multi-document RAG: silently averaging or picking
one fact would misinform the user. Per the spec, we NEVER merge
contradictory facts -- we surface them explicitly and recommend human
verification instead.

Only runs when 2+ distinct source documents are present in the retrieval,
since contradiction requires at least two independent sources.
"""

import logging

from models.schemas import RetrievalResult, ConflictReport, Conflict
from services.llm_client import generate_json

logger = logging.getLogger("raguard.contradiction_detector")

SYSTEM_INSTRUCTION = """You are a contradiction detection component in a document Q&A system.
You will be given a question and a numbered list of text chunks from possibly different documents.
Determine if any chunks make factually CONTRADICTORY claims relevant to the question
(e.g. different numbers, dates, or requirements for the same thing).
Do NOT flag chunks that are simply about different subtopics or that complement each other.

Return ONLY a JSON object with this exact key:
- conflicts: a list of objects, each with:
  - chunk_number_a (integer)
  - chunk_number_b (integer)
  - statement_a (the conflicting claim from chunk A, in your own words, one sentence)
  - statement_b (the conflicting claim from chunk B, in your own words, one sentence)
  - explanation (why these conflict, one sentence)

If there are no contradictions, return {"conflicts": []}."""


def detect_contradictions(query: str, retrieval: RetrievalResult) -> ConflictReport:
    """Compare retrieved chunks for factual contradictions between sources."""
    distinct_documents = {r.chunk.document_id for r in retrieval.results}

    if len(distinct_documents) < 2:
        # Contradiction requires independent sources -- skip the LLM call
        # entirely when everything came from one document.
        return ConflictReport(has_conflict=False, conflicts=[], recommendation="None")

    numbered_chunks = "\n\n".join(
        f"[Chunk {i+1}] (doc: {r.chunk.document_name})\n{r.chunk.text}"
        for i, r in enumerate(retrieval.results)
    )
    prompt = f'Question: "{query}"\n\nChunks:\n{numbered_chunks}'

    result = generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)
    raw_conflicts = result.get("conflicts", [])

    if not raw_conflicts:
        return ConflictReport(has_conflict=False, conflicts=[], recommendation="None")

    conflicts: list[Conflict] = []
    for c in raw_conflicts:
        a_idx, b_idx = c.get("chunk_number_a"), c.get("chunk_number_b")
        if not a_idx or not b_idx:
            continue
        if not (1 <= a_idx <= len(retrieval.results)) or not (1 <= b_idx <= len(retrieval.results)):
            continue

        chunk_a = retrieval.results[a_idx - 1].chunk
        chunk_b = retrieval.results[b_idx - 1].chunk

        conflicts.append(
            Conflict(
                chunk_id_a=chunk_a.chunk_id,
                document_a=chunk_a.document_name,
                statement_a=c.get("statement_a", ""),
                chunk_id_b=chunk_b.chunk_id,
                document_b=chunk_b.document_name,
                statement_b=c.get("statement_b", ""),
                explanation=c.get("explanation", ""),
            )
        )

    if conflicts:
        logger.warning(f"Detected {len(conflicts)} contradiction(s) for query: {query}")

    return ConflictReport(
        has_conflict=len(conflicts) > 0,
        conflicts=conflicts,
        recommendation="Human verification required" if conflicts else "None",
    )