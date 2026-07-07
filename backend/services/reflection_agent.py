"""
Reflection Agent (spec Agent 6).

Runs AFTER evidence verification passes, as one final self-check before
committing to an answer. This is deliberately a separate step from
verification: verification asks "does this evidence answer the question at
all?" while reflection asks the more holistic question "is this the BEST
I can do, or would one more retrieval round meaningfully improve things?"

Per the spec's loop diagram, if reflection says another retrieval would
help, the loop goes around once more -- but we cap this at one reflection-
triggered retry to avoid infinite loops eating into latency/cost, since
the quality+verification gates already did the heavy lifting.
"""

import logging

from models.schemas import RetrievalResult, EvidenceVerification, ReflectionResult
from services.llm_client import generate_json

logger = logging.getLogger("raguard.reflection")

SYSTEM_INSTRUCTION = """You are a reflection component in a document Q&A system, acting as a
cautious internal reviewer before an answer is shown to a user.

Given a question, the retrieved evidence, and a prior verification verdict, reflect on:
- Did the evidence actually answer the question well, or just partially?
- Is any clearly important piece of evidence likely missing (e.g. the chunks mention
  a policy exists but not its exact value, or reference something not retrieved)?
- Would retrieving again with a different angle plausibly surface better evidence?

Return ONLY a JSON object with these exact keys:
- is_satisfied: true/false -- are you satisfied this evidence is enough to answer well?
- missing_evidence: a short description of what's missing, or null if nothing missing
- should_retrieve_again: true/false -- only true if retrieving again would plausibly help
  (false if the evidence is simply absent from the documents entirely, since retrying
  won't invent evidence that isn't there)
- reasoning: one or two sentence explanation

Be conservative with should_retrieve_again: only recommend it if there's a concrete,
different angle worth trying, not just because the answer feels incomplete."""


def reflect_on_answer(
    query: str,
    retrieval: RetrievalResult,
    verification: EvidenceVerification,
) -> ReflectionResult:
    """Run the Reflection Agent's post-verification self-check."""
    supporting_texts = "\n\n".join(
        f"- {r.chunk.text}"
        for r in retrieval.results
        if r.chunk.chunk_id in verification.supporting_chunk_ids
    ) or "\n\n".join(f"- {r.chunk.text}" for r in retrieval.results)

    prompt = f"""Question: "{query}"

Supporting evidence found:
{supporting_texts}

Verification verdict: answers_question={verification.answers_question}, confidence={verification.confidence}
Verification reasoning: {verification.reasoning}
"""

    result = generate_json(prompt, system_instruction=SYSTEM_INSTRUCTION)

    if not result:
        logger.warning("Reflection LLM call failed, defaulting to satisfied (fail-open)")
        # Fail-open here (not fail-closed): if reflection itself breaks, we'd
        # rather return the already-verified answer than loop forever.
        return ReflectionResult(is_satisfied=True, should_retrieve_again=False, reasoning="Reflection unavailable.")

    return ReflectionResult(
        is_satisfied=result.get("is_satisfied", True),
        missing_evidence=result.get("missing_evidence"),
        should_retrieve_again=result.get("should_retrieve_again", False),
        reasoning=result.get("reasoning", ""),
    )