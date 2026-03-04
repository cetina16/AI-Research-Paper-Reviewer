"""
Review quality evaluation (LLM-as-judge + rule-based checks).

Evaluates a generated review on five dimensions:
  1. Groundedness   — criticisms based on actual paper content?
  2. Hallucination  — non-existent claims introduced?
  3. Reference validity — cited related works real and relevant?
  4. Completeness   — all expected sections present?
  5. Structure compliance — matches the target venue template?

Also performs lightweight rule-based checks (section presence, score
range, recommendation consistency) before calling Claude.
"""
from __future__ import annotations

import re

import anthropic
from rich.console import Console

from src.schemas import (
    EvaluationResult,
    PaperComponents,
    RelatedPaper,
    StructuredReview,
)
from src.retrieval import _build_schema

console = Console()

# Sections expected in a well-formed review
_REQUIRED_SECTIONS = [
    "summary",
    "strengths",
    "weaknesses",
    "questions",
    "limitations",
    "novelty_assessment",
    "technical_quality",
    "score",
    "recommendation",
    "justification",
]

_VALID_RECOMMENDATIONS = {
    "Strong Accept",
    "Accept",
    "Weak Accept",
    "Borderline",
    "Weak Reject",
    "Reject",
    "Strong Reject",
}


# ---------------------------------------------------------------------------
# Rule-based pre-checks
# ---------------------------------------------------------------------------

def _rule_based_checks(review: StructuredReview) -> list[str]:
    """Return a list of rule-violation strings (empty = all good)."""
    issues: list[str] = []

    # Minimum content length checks
    if len(review.summary.split()) < 30:
        issues.append("Summary is very short (< 30 words).")
    if len(review.strengths) < 1:
        issues.append("No strengths listed.")
    if len(review.weaknesses) < 1:
        issues.append("No weaknesses listed.")
    if len(review.questions) < 1:
        issues.append("No questions listed for the authors.")

    # Score / recommendation consistency
    rec = review.recommendation
    score = review.score
    if rec in ("Strong Accept", "Accept") and score < 6:
        issues.append(
            f"Recommendation '{rec}' inconsistent with score {score}/10."
        )
    if rec in ("Reject", "Strong Reject") and score > 5:
        issues.append(
            f"Recommendation '{rec}' inconsistent with score {score}/10."
        )
    if rec not in _VALID_RECOMMENDATIONS:
        issues.append(f"Unrecognised recommendation: '{rec}'.")

    # Score range
    if not (1 <= score <= 10):
        issues.append(f"Score {score} is outside the valid range 1–10.")
    if not (1 <= review.confidence <= 5):
        issues.append(f"Confidence {review.confidence} is outside the valid range 1–5.")

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_review(
    review: StructuredReview,
    components: PaperComponents,
    paper_text: str,
    related_papers: list[RelatedPaper],
    client: anthropic.Anthropic,
) -> EvaluationResult:
    """
    Run a full quality evaluation of the review.

    1. Rule-based checks (fast).
    2. LLM-as-judge evaluation (Claude).

    Returns an EvaluationResult with scores and specific issues.
    """
    console.print("[cyan]Evaluating review quality…[/cyan]")

    rule_issues = _rule_based_checks(review)
    if rule_issues:
        console.print(f"[yellow]Rule-based issues found: {len(rule_issues)}[/yellow]")
        for iss in rule_issues:
            console.print(f"  [yellow]• {iss}[/yellow]")

    # Truncate paper text to avoid blowing the context window
    paper_snippet = paper_text[:20_000]
    review_json = review.model_dump_json(indent=2)

    related_titles = (
        "\n".join(f"  • {p.title} ({p.year or '?'})" for p in related_papers)
        or "  (none retrieved)"
    )

    rule_issues_str = (
        "\n".join(f"  • {i}" for i in rule_issues)
        if rule_issues
        else "  None detected."
    )

    schema = _build_schema(EvaluationResult)

    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=(
            "You are an expert meta-reviewer evaluating the quality of a peer review. "
            "Be rigorous, fair, and specific. Your job is to identify flaws in the review, "
            "not to re-review the paper itself."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "=== PAPER (excerpt) ===\n"
                    f"{paper_snippet}\n\n"
                    "=== PAPER KEY COMPONENTS ===\n"
                    f"Title: {components.title}\n"
                    f"Main claims: {'; '.join(components.main_claims[:5])}\n"
                    f"Key contributions: {'; '.join(components.key_contributions[:5])}\n"
                    f"Baselines: {', '.join(components.baselines_mentioned) or 'none'}\n"
                    f"Datasets: {', '.join(components.datasets_used) or 'none'}\n\n"
                    "=== RETRIEVED RELATED PAPERS (ground-truth for reference validation) ===\n"
                    f"{related_titles}\n\n"
                    "=== RULE-BASED PRE-CHECK ISSUES ===\n"
                    f"{rule_issues_str}\n\n"
                    "=== REVIEW TO EVALUATE ===\n"
                    f"{review_json}\n\n"
                    "Evaluate the review on these dimensions:\n\n"
                    "1. **Groundedness** (0–1): Are the strengths/weaknesses backed by actual "
                    "paper content? List any ungrounded claims.\n\n"
                    "2. **Hallucination rate** (0–1, 0=none): Does the review introduce facts, "
                    "results, or authors NOT in the paper? List examples.\n\n"
                    "3. **Reference validity** (0–1): Are related works cited in the review "
                    "real and present in the retrieved papers list? List invalid references.\n\n"
                    "4. **Completeness** (0–1): Does the review cover summary, strengths, "
                    "weaknesses, questions, limitations, novelty, technical quality, "
                    "and scoring? List missing sections.\n\n"
                    "5. **Structure compliance** (0–1): Does the review match the expected "
                    "venue format? List structural issues.\n\n"
                    "6. **Overall quality** (0–1): Weighted combination of all above.\n\n"
                    "7. **Improvement suggestions**: Specific, actionable improvements.\n\n"
                    "Respond with the JSON evaluation."
                ),
            }
        ],
        output_format=EvaluationResult,
    )

    result: EvaluationResult = response.parsed_output

    console.print(
        f"[green]✓ Evaluation complete — Overall quality: "
        f"{result.overall_quality_score:.2f}[/green]"
    )
    _log_evaluation_summary(result)
    return result


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def _log_evaluation_summary(result: EvaluationResult) -> None:
    metrics = [
        ("Groundedness", result.groundedness_score),
        ("Hallucination rate", result.hallucination_rate),
        ("Reference validity", result.reference_validity_score),
        ("Completeness", result.completeness_score),
        ("Structure compliance", result.structure_compliance_score),
        ("Overall quality", result.overall_quality_score),
    ]
    for name, val in metrics:
        colour = "green" if val >= 0.7 else ("yellow" if val >= 0.4 else "red")
        # For hallucination rate, lower is better
        if name == "Hallucination rate":
            colour = "green" if val <= 0.1 else ("yellow" if val <= 0.3 else "red")
        console.print(f"  [{colour}]{name}: {val:.2f}[/{colour}]")
