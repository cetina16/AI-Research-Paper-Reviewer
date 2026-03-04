"""
Pydantic schemas for the AI Research Paper Reviewer.
All Claude-output models use ConfigDict(extra='forbid') to ensure
proper additionalProperties:false in JSON schema for structured outputs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Paper Extraction
# ---------------------------------------------------------------------------

class PaperComponents(BaseModel):
    """Key components extracted from a research paper."""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Paper title")
    authors: List[str] = Field(description="Author names")
    abstract: str = Field(description="Full abstract text")
    problem_statement: str = Field(description="Core problem the paper addresses")
    methodology: str = Field(description="Methods and technical approaches")
    experiments: str = Field(description="Description of experiments conducted")
    results: str = Field(description="Key results and findings")
    main_claims: List[str] = Field(description="Explicit claims made in the paper")
    baselines_mentioned: List[str] = Field(description="Baselines compared against (empty list if none)")
    datasets_used: List[str] = Field(description="Datasets used in experiments (empty list if none)")
    key_contributions: List[str] = Field(description="Main contributions stated by authors")
    limitations_mentioned: List[str] = Field(description="Limitations acknowledged by authors (empty list if none)")


# ---------------------------------------------------------------------------
# Related Work Retrieval
# ---------------------------------------------------------------------------

class RelatedPaper(BaseModel):
    """A related paper found via retrieval."""
    model_config = ConfigDict(extra="forbid")

    title: str
    authors: List[str]
    year: Optional[int] = None
    abstract: Optional[str] = None
    citation_count: Optional[int] = None
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance 0-1")
    relevance_reason: str = Field(description="Why this paper is relevant")


class RelatedPapersRanking(BaseModel):
    """Ranked list of related papers (internal use for parsing)."""
    model_config = ConfigDict(extra="forbid")

    papers: List[RelatedPaper]


class NoveltyAssessment(BaseModel):
    """Assessment of paper novelty relative to prior work."""
    model_config = ConfigDict(extra="forbid")

    novelty_level: str = Field(
        description="One of: high / medium / low / incremental"
    )
    novel_aspects: List[str] = Field(description="Genuinely novel contributions")
    similar_prior_work: List[str] = Field(
        description="Titles of similar prior works (from retrieved papers)"
    )
    overlap_concerns: List[str] = Field(
        description="Areas of significant overlap with prior work"
    )
    novelty_justification: str = Field(description="Overall novelty justification")


# ---------------------------------------------------------------------------
# Structured Review
# ---------------------------------------------------------------------------

class StructuredReview(BaseModel):
    """A complete structured research paper review."""
    model_config = ConfigDict(extra="forbid")

    venue: str = Field(description="Target venue (NeurIPS, ICML, ICLR, etc.)")
    paper_title: str
    summary: str = Field(description="1–2 paragraph summary of the paper")
    strengths: List[str] = Field(description="Clear strengths (at least 2)")
    weaknesses: List[str] = Field(description="Clear weaknesses and concerns (at least 2)")
    questions: List[str] = Field(description="Clarifying questions for the authors")
    limitations: List[str] = Field(
        description="Identified limitations, including those not acknowledged by authors"
    )
    ethical_concerns: str = Field(
        default="None identified.",
        description="Ethical concerns, or 'None identified.' if none"
    )
    novelty_assessment: str = Field(description="Assessment of novelty and originality")
    technical_quality: str = Field(description="Assessment of technical rigour")
    related_works_missing: List[str] = Field(
        description="Important related works not cited (empty list if none)"
    )
    unsupported_claims: List[str] = Field(
        description="Claims not adequately supported by experiments/theory (empty list if none)"
    )
    score: int = Field(
        description=(
            "Overall score 1–10: "
            "1=Very Strong Reject, 3=Reject, 5=Borderline, "
            "7=Accept, 9=Very Strong Accept"
        )
    )
    confidence: int = Field(
        description="Reviewer confidence 1–5 (1=not sure, 5=very confident)"
    )
    recommendation: str = Field(
        description="One of: Strong Accept / Accept / Weak Accept / Borderline / Weak Reject / Reject / Strong Reject"
    )
    justification: str = Field(description="Overall justification for the score")


# ---------------------------------------------------------------------------
# Review Evaluation (LLM-as-Judge)
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Quality assessment of a generated review."""
    model_config = ConfigDict(extra="forbid")

    groundedness_score: float = Field(
        ge=0.0, le=1.0,
        description="How well criticisms/strengths are grounded in actual paper content (1=fully grounded)"
    )
    groundedness_issues: List[str] = Field(
        description="Review statements not grounded in the paper (empty if none)"
    )
    hallucination_rate: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of review content that introduces non-existent claims (0=no hallucination)"
    )
    hallucination_examples: List[str] = Field(
        description="Specific hallucinated statements found (empty if none)"
    )
    reference_validity_score: float = Field(
        ge=0.0, le=1.0,
        description="How valid/relevant cited related works are (1=all valid)"
    )
    invalid_references: List[str] = Field(
        description="References that seem invalid or irrelevant (empty if none)"
    )
    completeness_score: float = Field(
        ge=0.0, le=1.0,
        description="How complete the review is (1=covers all required aspects)"
    )
    missing_sections: List[str] = Field(
        description="Expected review aspects that are missing or superficial (empty if none)"
    )
    structure_compliance_score: float = Field(
        ge=0.0, le=1.0,
        description="Adherence to the target venue review template (1=fully compliant)"
    )
    overall_quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Weighted overall review quality (1=excellent)"
    )
    improvement_suggestions: List[str] = Field(
        description="Specific suggestions to improve the review"
    )


# ---------------------------------------------------------------------------
# Monitoring & Adaptive Config
# ---------------------------------------------------------------------------

class ReviewMetrics(BaseModel):
    """Persisted metrics for one review run."""
    review_id: str
    timestamp: str
    paper_title: str
    venue: str
    model_used: str
    retrieval_depth: int
    prompt_variant: str
    processing_time_seconds: float
    groundedness_score: float
    hallucination_rate: float
    reference_validity_score: float
    completeness_score: float
    structure_compliance_score: float
    overall_quality_score: float
    review_score: int
    review_confidence: int
    second_pass_used: bool


class AdaptiveConfig(BaseModel):
    """Runtime configuration adjusted by the adaptive controller."""
    model: str = "claude-opus-4-6"
    retrieval_depth: int = Field(default=5, description="Related papers to retrieve")
    prompt_variant: str = Field(
        default="standard",
        description="Prompt style: standard | detailed | constrained"
    )
    enable_second_pass: bool = Field(
        default=False,
        description="Enable a second-pass critique for low-confidence reviews"
    )
    max_tokens: int = Field(default=8192)
    adjustments_made: List[str] = Field(
        default_factory=list,
        description="Human-readable list of adjustments applied"
    )
