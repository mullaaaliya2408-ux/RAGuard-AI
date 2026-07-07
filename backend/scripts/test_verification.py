"""
Standalone script to test evidence verification and contradiction
detection directly, independent of the full self-correction loop.

Usage:
    python scripts/test_verification.py "your question"
"""

import sys

from services.hybrid_retriever import hybrid_search
from services.evidence_verifier import verify_evidence
from services.contradiction_detector import detect_contradictions


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_verification.py "your question"')
        sys.exit(1)

    query = sys.argv[1]
    retrieval = hybrid_search(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(retrieval.results)} chunks\n")

    verification = verify_evidence(query, retrieval)
    print("--- Evidence Verification ---")
    print(f"Answers question: {verification.answers_question}")
    print(f"Confidence: {verification.confidence}")
    print(f"Reasoning: {verification.reasoning}")
    print(f"Supporting chunks: {len(verification.supporting_chunk_ids)}")
    print(f"Unsupporting chunks: {len(verification.unsupporting_chunk_ids)}\n")

    conflict_report = detect_contradictions(query, retrieval)
    print("--- Contradiction Report ---")
    print(f"Has conflict: {conflict_report.has_conflict}")
    for c in conflict_report.conflicts:
        print(f"  [{c.document_a}] {c.statement_a}")
        print(f"  [{c.document_b}] {c.statement_b}")
        print(f"  Why: {c.explanation}\n")
    print(f"Recommendation: {conflict_report.recommendation}")


if __name__ == "__main__":
    main()