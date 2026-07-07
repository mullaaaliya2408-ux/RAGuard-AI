"""
Evidence Verification Agent (spec Agent 4).

A chunk can pass the similarity/BM25 threshold in Milestone 5 while still
not actually containing the answer -- e.g. it's about the right *topic*
but the specific fact asked for isn't in it. This agent makes an explicit
LLM judgment: "given these chunks, can this question actually be answered?"

This is deliberately separate from the Retrieval Quality Checker (Milestone
5), which is a cheap, deterministic numeric gate. This agent is a slower,
more expensive semantic judgment call, so the loop only invokes it AFTER
the numeric gate has already passed -- no point spending an LLM call
verifying evidence that failed a cheap check first.
"""

import logging

from models.schemas import RetrievalResult, EvidenceVerification
from services.llm_client import generate_json

logger = logging.getLogger("raguard.evidence_verifier")

SYSTEM_INSTRUCTION = """You are an evidence verification component in a document Q&A system.
You will be given a question and a numbered list of retrieved text chunks.
Your job is to judge whether the chunks, taken together, actually contain enough
information to answer the question -- not just whether they are on a related topic.

Return ONLY a JSON object with these exact keys:
- answers_question: true/false
- confidence: a number 0-1 representing how confident you are in this verdict
- supporting_chunk_numbers: list of chunk numbers (integers) that genuinely help answer the question
- unsupporting_chunk_numbers: list of chunk numbers that are related but do NOT help answer it
- reasoning: one or two sentence explanation of your verdict

Be strict: if the chunks only mention the general topic without the specific fact asked for,
answers_question should be false."""


def verify_evidence(query: str, retrieval: RetrievalResult) -> EvidenceVerification:
    """Ask the LLM whether the retrieved chunks actually answer the question."""
    if not retrieval.results:
        return EvidenceVerification(
            answers_question=False,
            confidence=1.0,
            reasoning="No chunks were retrieved.",
        )

    numbered_chunks = "\n\n".join(
        f"[Chunk {i+1}] (page {r.chunk.page_number}, doc: {r.chunk.document_name})\n{r.chunk.text}"
        for i, r in enumerate(retrieval.results)
    )
    prompt = f'Question: "{query}"\n\nRetrieved chunks:\n{numbered_chunks}'

    result = generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)

    if not result:
        logger.warning("Evidence verification LLM call failed, defaulting to unverified")
        return EvidenceVerification(
            answers_question=False,
            confidence=0.0,
            reasoning="Verification could not be completed due to an LLM error.",
        )

    # Map chunk numbers (1-indexed, as shown to the LLM) back to real chunk_ids
    def numbers_to_chunk_ids(numbers: list[int]) -> list[str]:
        ids = []
        for n in numbers:
            if 1 <= n <= len(retrieval.results):
                ids.append(retrieval.results[n - 1].chunk.chunk_id)
        return ids

    return EvidenceVerification(
        answers_question=result.get("answers_question", False),
        confidence=result.get("confidence", 0.0),
        supporting_chunk_ids=numbers_to_chunk_ids(result.get("supporting_chunk_numbers", [])),
        unsupporting_chunk_ids=numbers_to_chunk_ids(result.get("unsupporting_chunk_numbers", [])),
        reasoning=result.get("reasoning", ""),
    )