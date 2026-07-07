"""
15 benchmark questions used by the Evaluation page (spec: EVALUATION PAGE).

These are intentionally generic phrasings about common document types
(policies, procedures, definitions) since we don't know in advance what
the user has uploaded. A production version would let users define their
own benchmark set per document collection -- noted as a future improvement.

expected_answerable=True means we expect the corpus to contain evidence
(used to measure accuracy/hallucination rate); False means we expect these
to correctly trigger refusal (used to measure over-eager hallucination).
"""

from pydantic import BaseModel


class BenchmarkQuestion(BaseModel):
    id: int
    question: str
    expected_answerable: bool


BENCHMARK_QUESTIONS: list[BenchmarkQuestion] = [
    BenchmarkQuestion(id=1, question="What is the minimum attendance percentage required?", expected_answerable=True),
    BenchmarkQuestion(id=2, question="What is the policy on late submissions?", expected_answerable=True),
    BenchmarkQuestion(id=3, question="What documents are required for admission?", expected_answerable=True),
    BenchmarkQuestion(id=4, question="What is the grading scale used?", expected_answerable=True),
    BenchmarkQuestion(id=5, question="What is the procedure for requesting a leave of absence?", expected_answerable=True),
    BenchmarkQuestion(id=6, question="What are the consequences of academic dishonesty?", expected_answerable=True),
    BenchmarkQuestion(id=7, question="What is the refund policy for withdrawn courses?", expected_answerable=True),
    BenchmarkQuestion(id=8, question="Who should be contacted for technical support?", expected_answerable=True),
    BenchmarkQuestion(id=9, question="What is the deadline for fee payment?", expected_answerable=True),
    BenchmarkQuestion(id=10, question="What is the dress code policy?", expected_answerable=True),
    BenchmarkQuestion(id=11, question="What is the capital of France?", expected_answerable=False),
    BenchmarkQuestion(id=12, question="What is the current stock price of Apple?", expected_answerable=False),
    BenchmarkQuestion(id=13, question="Who won the most recent World Cup?", expected_answerable=False),
    BenchmarkQuestion(id=14, question="What is the airspeed velocity of an unladen swallow?", expected_answerable=False),
    BenchmarkQuestion(id=15, question="What is the boiling point of mercury?", expected_answerable=False),
]