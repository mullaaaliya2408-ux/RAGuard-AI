"""
Shared data structures used across the ingestion pipeline.
Keeping these in one place means every service speaks the same "language".
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class DocumentType(str, Enum):
    DIGITAL_PDF = "digital_pdf"
    SCANNED_PDF = "scanned_pdf"
    IMAGE = "image"
    MIXED_PDF = "mixed_pdf"


class PageResult(BaseModel):
    """Extracted content for a single page."""
    page_number: int
    text: str
    ocr_confidence: Optional[float] = None  # None for digital text, 0-1 for OCR
    used_ocr: bool = False


class DocumentMetadata(BaseModel):
    """Metadata returned to the frontend after processing a document."""
    document_id: str
    filename: str
    document_type: DocumentType
    total_pages: int
    total_characters: int
    average_ocr_confidence: Optional[float] = None
    processing_warnings: list[str] = []
class Chunk(BaseModel):
    """
    A single retrievable unit of text with full provenance metadata.
    Every chunk can be traced back to its exact document, page, and position.
    """
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    section: Optional[str] = None
    text: str
    chunk_length: int
    ocr_confidence: Optional[float] = None
    chunk_index: int  # position within the document, for ordering    
class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, annotated with scoring info."""
    chunk: "Chunk"
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    metadata_score: float = 0.0
    combined_score: float = 0.0
    matched_via: list[str] = []  # e.g. ["semantic", "bm25"]


class RetrievalResult(BaseModel):
    """Full output of a retrieval pass, used by later quality-check agents."""
    query: str
    expanded_queries: list[str]
    results: list[RetrievedChunk]
    average_similarity: float = 0.0 
class QueryAnalysis(BaseModel):
    """Structured understanding of a user's query, produced before retrieval."""
    original_query: str
    intent: str  # e.g. "policy_lookup", "definition", "comparison", "procedure"
    question_type: str  # e.g. "factual", "yes_no", "how_to", "comparison"
    entities: list[str] = []
    keywords: list[str] = []
    document_references: list[str] = []  # explicit doc names mentioned, if any
    rewritten_query: Optional[str] = None  # cleaned-up query, if ambiguous
    is_ambiguous: bool = False   
class QualityAssessment(BaseModel):
    """Output of the Retrieval Quality Checker for a single retrieval pass."""
    average_similarity: float
    coverage_score: float       # how many distinct sections/pages are represented
    evidence_diversity: float   # how many distinct documents contributed
    metadata_quality: float     # fraction of chunks with section/OCR info intact
    overall_score: float
    passed: bool
    reasons: list[str] = []
class RetrievalAttempt(BaseModel):
    """Record of a single retrieval attempt within the self-correction loop."""
    query_used: str
    quality: QualityAssessment
    was_rewrite: bool


class SelfCorrectionResult(BaseModel):
    """Full output of the self-correction loop, ready for downstream agents."""
    original_query: str
    query_analysis: QueryAnalysis
    final_retrieval: RetrievalResult
    final_quality: QualityAssessment
    attempts: list[RetrievalAttempt]
    succeeded: bool
class EvidenceVerification(BaseModel):
    """Output of the Evidence Verification Agent for one retrieval attempt."""
    answers_question: bool
    confidence: float  # 0-1, the agent's own confidence in its verdict
    supporting_chunk_ids: list[str] = []  # chunks that actually contain the answer
    unsupporting_chunk_ids: list[str] = []
    reasoning: str = ""


class Conflict(BaseModel):
    """A single detected contradiction between two chunks."""
    chunk_id_a: str
    document_a: str
    statement_a: str
    chunk_id_b: str
    document_b: str
    statement_b: str
    explanation: str


class ConflictReport(BaseModel):
    """Full contradiction check result for a retrieval attempt."""
    has_conflict: bool
    conflicts: list[Conflict] = []
    recommendation: str = "None"  # e.g. "Human verification required"


class ReflectionResult(BaseModel):
    """Output of the Reflection Agent's post-verification self-check."""
    is_satisfied: bool
    missing_evidence: Optional[str] = None
    should_retrieve_again: bool
    reasoning: str = ""

class ConfidenceFactors(BaseModel):
    """
    Individual signals contributing to the final confidence score.
    """
    similarity_score: float
    verification_confidence: float
    evidence_agreement: float
    supporting_chunk_count_score: float
    chunk_quality_score: float
    ocr_confidence_score: float
    retrieval_success_score: float


class ConfidenceReport(BaseModel):
    """
    User-facing confidence report.
    """
    score: float
    label: str
    factors: ConfidenceFactors
    explanation: str


class FinalAnswer(BaseModel):
    """
    The complete, user-facing response format required by the spec:
    Final Answer, Confidence Score, Evidence Sources, Retrieved Chunks,
    Reasoning Summary, Warnings, Suggested Follow-up.
    """
    final_answer: str
    confidence: "ConfidenceReport"
    evidence_sources: list[str] = []
    retrieved_chunk_ids: list[str] = []
    reasoning_summary: str = ""
    warnings: list[str] = []
    suggested_follow_up: Optional[str] = None

class PipelineResult(BaseModel):
    """
    Complete output of the full pipeline: self-correction loop +
    reflection + final answer. This is what the API returns.
    """
    self_correction: SelfCorrectionResult
    reflection: Optional[ReflectionResult] = None
    final_answer: FinalAnswer
class BenchmarkResult(BaseModel):
    """Result of running one benchmark question through the pipeline."""
    question_id: int
    question: str
    expected_answerable: bool
    was_answered: bool          # did the system provide a substantive answer (not a refusal)?
    is_correct: bool            # was_answered matches expected_answerable
    confidence_score: float
    confidence_label: str
    latency_ms: float
    attempts_used: int


class EvaluationSummary(BaseModel):
    """Aggregate metrics across all benchmark questions, for the Evaluation page."""
    total_questions: int
    accuracy: float               # fraction where is_correct
    hallucination_rate: float     # fraction of expected_answerable=False that were answered anyway
    precision: float              # of answered questions, fraction expected_answerable=True
    recall: float                 # of expected_answerable=True questions, fraction actually answered
    average_latency_ms: float
    average_confidence: float
    confidence_distribution: dict[str, int]  # {"High": n, "Medium": n, "Low": n}
    results: list[BenchmarkResult]