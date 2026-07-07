"""
Confidence Engine (spec section: CONFIDENCE ENGINE).

Combines every signal already computed elsewhere in the pipeline into one
transparent, weighted confidence score:
  - Similarity score          (from hybrid retrieval, Milestone 4)
  - Evidence agreement        (from contradiction detector, Milestone 6)
  - Supporting chunk count    (from evidence verifier, Milestone 6)
  - Chunk quality             (metadata quality, from quality checker, Milestone 5)
  - OCR confidence            (inherited from document ingestion, Milestone 2)
  - Retrieval success         (did self-correction succeed cleanly, or limp
                                across the finish line on best-effort?)
  - Verification confidence   (the Evidence Verification Agent's own judgment)

Deliberately rule-based, not an LLM call: confidence scoring needs to be
fast, deterministic, and fully explainable -- if a user asks "why is this
Low confidence?", we need a precise numeric answer, not another model's
opinion about its own opinion.
"""

import logging

from models.schemas import (
    RetrievalResult,
    QualityAssessment,
    EvidenceVerification,
    ConflictReport,
    ConfidenceFactors,
    ConfidenceReport,
)

logger = logging.getLogger("raguard.confidence_engine")

# Weights sum to 1.0. Similarity and verification confidence are weighted
# highest since they most directly reflect "does the evidence answer this".
WEIGHTS = {
    "similarity_score": 0.25,
    "verification_confidence": 0.25,
    "evidence_agreement": 0.15,
    "supporting_chunk_count_score": 0.15,
    "chunk_quality_score": 0.10,
    "ocr_confidence_score": 0.05,
    "retrieval_success_score": 0.05,
}

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.5


def compute_evidence_agreement(conflict_report: ConflictReport | None) -> float:
    """Full agreement (1.0) if no conflicts found; penalized per conflict otherwise."""
    if conflict_report is None or not conflict_report.has_conflict:
        return 1.0
    penalty = min(len(conflict_report.conflicts) * 0.3, 0.8)
    return round(1.0 - penalty, 3)


def compute_supporting_chunk_score(verification: EvidenceVerification) -> float:
    """
    More independent supporting chunks = more confidence, but with
    diminishing returns -- 1 supporting chunk is fine, 3+ is comfortably
    strong, and beyond that adds little extra signal.
    """
    count = len(verification.supporting_chunk_ids)
    return min(count / 3, 1.0)


def compute_chunk_quality_score(retrieval: RetrievalResult) -> float:
    """
    Average metadata completeness across retrieved chunks: does each chunk
    have a known section, and (if OCR-derived) reasonable OCR confidence?
    Reuses the same logic style as the Milestone 5 quality checker, but
    scoped here to the confidence report rather than the retry gate.
    """
    if not retrieval.results:
        return 0.0
    scores = []
    for r in retrieval.results:
        score = 1.0
        if r.chunk.section is None:
            score -= 0.3
        scores.append(max(score, 0.0))
    return round(sum(scores) / len(scores), 3)


def compute_ocr_confidence_score(retrieval: RetrievalResult) -> float:
    """
    Average OCR confidence across chunks that came from OCR. Chunks with
    no OCR involvement (ocr_confidence is None, i.e. digital text) don't
    count against this score -- only genuinely OCR-derived evidence does.
    """
    ocr_scores = [r.chunk.ocr_confidence for r in retrieval.results if r.chunk.ocr_confidence is not None]
    if not ocr_scores:
        return 1.0  # no OCR involved at all -- nothing to penalize
    return round(sum(ocr_scores) / len(ocr_scores), 3)


def compute_retrieval_success_score(quality: QualityAssessment, succeeded: bool) -> float:
    """
    1.0 if the self-correction loop succeeded cleanly on a passing attempt;
    otherwise scaled down toward the raw quality score, since we're
    reporting a "best effort" result rather than a validated one.
    """
    if succeeded:
        return 1.0
    return round(quality.overall_score, 3)


def categorize(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def build_explanation(factors: ConfidenceFactors, label: str) -> str:
    """Produce a short, human-readable explanation of the dominant factors."""
    parts = []

    if factors.verification_confidence >= 0.7:
        parts.append("evidence verification was strongly confident")
    elif factors.verification_confidence < 0.4:
        parts.append("evidence verification had low confidence")

    if factors.evidence_agreement < 1.0:
        parts.append("conflicting information was found across sources")

    if factors.supporting_chunk_count_score < 0.34:
        parts.append("only a single supporting chunk was found")

    if factors.ocr_confidence_score < 0.6:
        parts.append("evidence relies on lower-quality OCR text")

    if factors.retrieval_success_score < 1.0:
        parts.append("retrieval did not fully clear quality thresholds")

    if not parts:
        parts.append("all supporting signals were strong and consistent")

    return f"{label} confidence: " + "; ".join(parts) + "."


def compute_confidence(
    retrieval: RetrievalResult,
    quality: QualityAssessment,
    verification: EvidenceVerification,
    conflict_report: ConflictReport | None,
    self_correction_succeeded: bool,
) -> ConfidenceReport:
    """Main entry point: compute the full multi-factor confidence report."""
    factors = ConfidenceFactors(
        similarity_score=retrieval.average_similarity,
        evidence_agreement=compute_evidence_agreement(conflict_report),
        supporting_chunk_count_score=compute_supporting_chunk_score(verification),
        chunk_quality_score=compute_chunk_quality_score(retrieval),
        ocr_confidence_score=compute_ocr_confidence_score(retrieval),
        retrieval_success_score=compute_retrieval_success_score(quality, self_correction_succeeded),
        verification_confidence=verification.confidence,
    )

    score = sum(getattr(factors, key) * weight for key, weight in WEIGHTS.items())
    score = round(min(max(score, 0.0), 1.0), 3)
    label = categorize(score)
    explanation = build_explanation(factors, label)

    logger.info(f"Confidence computed: {score} ({label})")

    return ConfidenceReport(score=score, label=label, factors=factors, explanation=explanation)