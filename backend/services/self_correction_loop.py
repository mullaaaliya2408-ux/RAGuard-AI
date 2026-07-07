"""
Self-Correction Loop controller (spec diagram: Retrieve -> Evaluate ->
Low Confidence? -> Rewrite -> Retrieve Again -> Verify Evidence -> Reflect
-> Satisfied? -> Answer).

Milestone 5 built: Retrieve -> Evaluate -> Rewrite -> Retry.
Milestone 6 adds: Verify Evidence as a second real gate, plus a one-time
contradiction check on whatever final retrieval we settle on.

Gate order per attempt:
  1. Quality check (cheap, deterministic) -- from Milestone 5
  2. Evidence verification (LLM judgment) -- only runs if (1) passed,
     since there's no point verifying evidence that failed the cheap
     check first
Both must pass for an attempt to be considered "succeeded". If either
fails, we rewrite and retry (up to MAX_ATTEMPTS).

Reflection (spec Agent 6) and Answer Generation (spec Agent 7) build on
top of this in the next milestones.
"""

import logging

from models.schemas import (

    SelfCorrectionResult,

    RetrievalAttempt,

    ReflectionResult,

)
from services.query_understanding import analyze_query
from services.hybrid_retriever import hybrid_search
from services.retrieval_quality_checker import assess_retrieval_quality
from services.query_rewriter import rewrite_query
from services.evidence_verifier import verify_evidence
from services.contradiction_detector import detect_contradictions
from services.reflection_agent import reflect_on_answer

logger = logging.getLogger("raguard.self_correction")

MAX_ATTEMPTS = 3
MIN_VERIFICATION_CONFIDENCE = 0.5


def run_self_correction_loop(query: str) -> tuple[SelfCorrectionResult, ReflectionResult | None]:
    """Main entry point: run the full retrieve -> evaluate -> verify -> rewrite cycle."""
    analysis = analyze_query(query)
    current_query = analysis.rewritten_query if analysis.is_ambiguous and analysis.rewritten_query else query

    attempts: list[RetrievalAttempt] = []
    best_result = None
    best_quality = None
    best_verification = None

    for attempt_number in range(1, MAX_ATTEMPTS + 1):
        logger.info(f"Attempt {attempt_number}: retrieving for '{current_query}'")

        retrieval = hybrid_search(current_query)
        quality = assess_retrieval_quality(retrieval)

        verification = None
        attempt_succeeded = False

        if quality.passed:
            verification = verify_evidence(current_query, retrieval)
            attempt_succeeded = (
                verification.answers_question
                and verification.confidence >= MIN_VERIFICATION_CONFIDENCE
            )

        attempts.append(
            RetrievalAttempt(
                query_used=current_query,
                quality=quality,
                was_rewrite=(attempt_number > 1),
            )
        )

        # Track best-so-far using quality score, but only "upgrade" our
        # best verification alongside it so the two stay consistent.
        if best_quality is None or quality.overall_score > best_quality.overall_score:
            best_result, best_quality, best_verification = retrieval, quality, verification

        if attempt_succeeded:
            logger.info(f"Quality + evidence verification both passed on attempt {attempt_number}")

            reflection = reflect_on_answer(current_query, retrieval, verification)

            # Reflection gets exactly one extra retry chance, separate from
            # MAX_ATTEMPTS, so a "should_retrieve_again" verdict doesn't loop
            # forever if it keeps firing.
            if reflection.should_retrieve_again and attempt_number < MAX_ATTEMPTS:
                logger.info(f"Reflection requested another retrieval: {reflection.reasoning}")
                rewrites = rewrite_query(current_query, analysis, quality)
                if rewrites:
                    current_query = rewrites[0]
                    continue  # loop again with the new query

            conflict_report = detect_contradictions(current_query, retrieval)
            return SelfCorrectionResult(
                original_query=query,
                query_analysis=analysis,
                final_retrieval=retrieval,
                final_quality=quality,
                final_verification=verification,
                conflict_report=conflict_report,
                attempts=attempts,
                succeeded=True,
            ), reflection
        
        if attempt_number < MAX_ATTEMPTS:
            # Reuse the quality checker's reasons for the rewrite prompt even
            # when the failure was actually verification-based, since the
            # rewriter's job either way is "find text that matches better".
            rewrite_reasons = quality.reasons if not quality.passed else (
                [verification.reasoning] if verification else []
            )
            quality_for_rewrite = quality.model_copy(update={"reasons": rewrite_reasons})
            rewrites = rewrite_query(current_query, analysis, quality_for_rewrite)
            if not rewrites:
                logger.warning("No rewrite candidates produced, stopping early")
                break
            current_query = rewrites[0]

    logger.info(f"Exhausted attempts without full success. Best quality score: {best_quality.overall_score}")
    conflict_report = detect_contradictions(query, best_result) if best_result else None

    return SelfCorrectionResult(
        original_query=query,
        query_analysis=analysis,
        final_retrieval=best_result,
        final_quality=best_quality,
        final_verification=best_verification,
        conflict_report=conflict_report,
        attempts=attempts,
        succeeded=False,
    ), None