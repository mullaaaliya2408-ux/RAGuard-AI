"""
Prints the full confidence factor breakdown for a query, so you can see
and explain exactly why a score came out High/Medium/Low.

Usage:
    python scripts/test_confidence.py "your question"
"""

import sys

from services.pipeline import run_pipeline


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/test_confidence.py "your question"')
        sys.exit(1)

    query = sys.argv[1]
    result = run_pipeline(query)
    confidence = result.final_answer.confidence

    print(f"\nQuestion: {query}")
    print(f"\nFinal Answer: {result.final_answer.final_answer}")

    print(f"\n{'='*50}")
    print(f"CONFIDENCE: {confidence.score} ({confidence.label})")
    print(f"{'='*50}")
    print(f"Explanation: {confidence.explanation}\n")

    print("Factor breakdown:")
    f = confidence.factors
    print(f"  Similarity score:              {f.similarity_score}")
    print(f"  Verification confidence:       {f.verification_confidence}")
    print(f"  Evidence agreement:            {f.evidence_agreement}")
    print(f"  Supporting chunk count score:  {f.supporting_chunk_count_score}")
    print(f"  Chunk quality score:           {f.chunk_quality_score}")
    print(f"  OCR confidence score:          {f.ocr_confidence_score}")
    print(f"  Retrieval success score:       {f.retrieval_success_score}")

    print(f"\nWarnings: {result.final_answer.warnings}")


if __name__ == "__main__":
    main()