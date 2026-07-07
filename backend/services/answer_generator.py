"""
Answer Generator (spec Agent 7).

Produces the final, user-facing answer in the exact structured format
required by the spec, now using the real Confidence Engine (Milestone 8)
instead of a placeholder blend. The LLM is instructed to answer using
ONLY the supplied evidence chunks -- this remains the last line of
defense against hallucination, on top of the numeric confidence gate.
"""

import logging

from models.schemas import (
    RetrievalResult,
    QualityAssessment,
    EvidenceVerification,
    ReflectionResult,
    ConflictReport,
    FinalAnswer,
)
from services.llm_client import generate_json
from services.confidence_engine import compute_confidence

logger = logging.getLogger("raguard.answer_generator")

SYSTEM_INSTRUCTION = """You are the answer-writing component of a cautious document Q&A system.
You will be given a question and a set of verified evidence chunks. Write a clear, direct
answer using ONLY the information in these chunks. Do not use outside knowledge. Do not guess
or fill gaps with assumptions.

If the evidence only partially covers the question, say so explicitly in the answer rather
than implying full certainty.

Return ONLY a JSON object with these exact keys:
- final_answer: the answer text, written for an end user, grounded strictly in the evidence
- reasoning_summary: one or two sentences on how you arrived at this answer from the evidence
- suggested_follow_up: a helpful follow-up question the user might ask next, or null"""

NO_EVIDENCE_ANSWER = "I could not find sufficient evidence to answer this question."


def generate_answer(
    query: str,
    retrieval: RetrievalResult,
    quality: QualityAssessment,
    verification: EvidenceVerification,
    reflection: ReflectionResult,
    conflict_report: ConflictReport | None,
    self_correction_succeeded: bool,
) -> FinalAnswer:
    """Generate the final structured answer, with a full confidence breakdown."""

    confidence = compute_confidence(
        retrieval=retrieval,
        quality=quality,
        verification=verification,
        conflict_report=conflict_report,
        self_correction_succeeded=self_correction_succeeded,
    )

    # Hallucination prevention gate: refuse outright if evidence doesn't
    # answer the question OR if the computed confidence lands in "Low" --
    # a Low label means we shouldn't present the answer as trustworthy,
    # regardless of whether the LLM could technically produce prose.
    if not verification.answers_question or not retrieval.results or confidence.label == "Low":
        return FinalAnswer(
            final_answer=NO_EVIDENCE_ANSWER,
            confidence=confidence,
            evidence_sources=[],
            retrieved_chunk_ids=[],
            reasoning_summary=verification.reasoning or "Insufficient evidence to answer confidently.",
            warnings=["Missing or low-confidence evidence for this question."],
            suggested_follow_up="Try rephrasing your question or checking if the relevant document has been uploaded.",
        )

    supporting_chunks = [
        r for r in retrieval.results if r.chunk.chunk_id in verification.supporting_chunk_ids
    ] or retrieval.results

    numbered_chunks = "\n\n".join(
        f"[Chunk {i+1}] (doc: {r.chunk.document_name}, page {r.chunk.page_number})\n{r.chunk.text}"
        for i, r in enumerate(supporting_chunks)
    )
    prompt = f'Question: "{query}"\n\nEvidence:\n{numbered_chunks}'

    result = generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)

    if not result:
        logger.error("Answer generation LLM call failed")
        return FinalAnswer(
            final_answer="An error occurred while generating the answer. Please try again.",
            confidence=confidence,
            warnings=["Answer generation failed."],
        )

    warnings: list[str] = []
    if reflection.missing_evidence:
        warnings.append(f"Possibly incomplete: {reflection.missing_evidence}")
    if conflict_report and conflict_report.has_conflict:
        warnings.append("Conflicting information found across sources — see contradiction report.")
    if confidence.label == "Medium":
        warnings.append("Confidence is Medium — evidence is reasonably strong but not fully conclusive.")

    evidence_sources = sorted({
        f"{r.chunk.document_name} (page {r.chunk.page_number})" for r in supporting_chunks
    })

    return FinalAnswer(
        final_answer=result.get("final_answer", NO_EVIDENCE_ANSWER),
        confidence=confidence,
        evidence_sources=evidence_sources,
        retrieved_chunk_ids=[r.chunk.chunk_id for r in supporting_chunks],
        reasoning_summary=result.get("reasoning_summary", ""),
        warnings=warnings,
        suggested_follow_up=result.get("suggested_follow_up"),
    )