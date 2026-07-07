"""
Retrieval Quality Checker (spec: "Retrieval Quality Checker" agent).

Scores a completed RetrievalResult across four dimensions and decides
whether it clears the bar to proceed to answer generation, or whether the
Self-Correction Loop should rewrite the query and retry.

This is a deterministic, rule-based checker (no LLM call) — quality
thresholds should be fast, cheap, and consistent, since they may run
multiple times per query inside the self-correction loop.
"""

import logging

from models.schemas import RetrievalResult, QualityAssessment

logger = logging.getLogger("raguard.quality_checker")

# Thresholds — tuned conservatively; can be exposed in Settings later
MIN_AVERAGE_SIMILARITY = 0.45
MIN_COVERAGE_SCORE = 0.3
MIN_OVERALL_SCORE = 0.45


def compute_coverage_score(result: RetrievalResult) -> float:
    """
    Coverage: how many distinct (document, page) locations are represented
    among the top results. Low coverage means all evidence comes from one
    narrow spot, which is risky if that spot happens to be wrong/irrelevant.
    """
    if not result.results:
        return 0.0
    locations = {(r.chunk.document_id, r.chunk.page_number) for r in result.results}
    # Normalize: 4+ distinct locations among top results counts as "full coverage"
    return min(len(locations) / 4, 1.0)


def compute_evidence_diversity(result: RetrievalResult) -> float:
    """How many distinct source documents contributed evidence."""
    if not result.results:
        return 0.0
    documents = {r.chunk.document_id for r in result.results}
    return min(len(documents) / 2, 1.0)  # 2+ documents = full diversity score


def compute_metadata_quality(result: RetrievalResult) -> float:
    """
    Fraction of top chunks that have clean supporting metadata: a known
    section AND (if OCR-derived) reasonable OCR confidence. Chunks with
    missing sections or poor OCR confidence lower trust in the evidence.
    """
    if not result.results:
        return 0.0

    scores = []
    for r in result.results:
        score = 1.0
        if r.chunk.section is None:
            score -= 0.3
        if r.chunk.ocr_confidence is not None and r.chunk.ocr_confidence < 0.6:
            score -= 0.4
        scores.append(max(score, 0.0))

    return sum(scores) / len(scores)


def assess_retrieval_quality(result: RetrievalResult) -> QualityAssessment:
    """Main entry point: compute all quality dimensions and decide pass/fail."""
    reasons: list[str] = []

    coverage = compute_coverage_score(result)
    diversity = compute_evidence_diversity(result)
    metadata_quality = compute_metadata_quality(result)

    if not result.results:
        reasons.append("No chunks were retrieved at all.")

    if result.average_similarity < MIN_AVERAGE_SIMILARITY:
        reasons.append(f"Average similarity {result.average_similarity} below threshold {MIN_AVERAGE_SIMILARITY}.")

    if coverage < MIN_COVERAGE_SCORE:
        reasons.append("Evidence is concentrated in very few locations — low coverage.")

    # Weighted overall score: similarity matters most, then coverage, then
    # diversity/metadata as secondary signals.
    overall = (
        0.5 * result.average_similarity
        + 0.25 * coverage
        + 0.15 * diversity
        + 0.10 * metadata_quality
    )

    passed = overall >= MIN_OVERALL_SCORE and len(result.results) > 0

    if not passed and not reasons:
        reasons.append("Overall retrieval quality score below threshold.")

    logger.info(f"Quality assessment: overall={round(overall, 3)} passed={passed}")

    return QualityAssessment(
        average_similarity=result.average_similarity,
        coverage_score=round(coverage, 3),
        evidence_diversity=round(diversity, 3),
        metadata_quality=round(metadata_quality, 3),
        overall_score=round(overall, 3),
        passed=passed,
        reasons=reasons,
    )