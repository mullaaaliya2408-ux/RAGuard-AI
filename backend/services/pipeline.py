"""
Top-level pipeline orchestrator: self-correction loop -> answer generation
with full confidence scoring. Single front door to the whole system.
"""

import logging

from models.schemas import PipelineResult, EvidenceVerification, ReflectionResult
from services.self_correction_loop import run_self_correction_loop
from services.answer_generator import generate_answer

logger = logging.getLogger("raguard.pipeline")


def run_pipeline(query: str) -> PipelineResult:
    """Run the full RAGuard pipeline for a user query and return the final answer."""
    self_correction, reflection = run_self_correction_loop(query)

    verification = self_correction.final_verification or EvidenceVerification(
        answers_question=False,
        confidence=0.0,
        reasoning="Evidence verification did not pass.",
    )

    effective_reflection = reflection or ReflectionResult(
        is_satisfied=self_correction.succeeded,
        should_retrieve_again=False,
        reasoning="",
    )

    final_answer = generate_answer(
        query=query,
        retrieval=self_correction.final_retrieval,
        quality=self_correction.final_quality,
        verification=verification,
        reflection=effective_reflection,
        conflict_report=self_correction.conflict_report,
        self_correction_succeeded=self_correction.succeeded,
    )

    return PipelineResult(
        self_correction=self_correction,
        reflection=reflection,
        final_answer=final_answer,
    )