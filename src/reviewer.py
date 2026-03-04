"""
Core review generation module.

Uses Claude Opus 4.6 with adaptive thinking to generate a structured,
venue-appropriate research paper review.  Three prompt variants are
supported (standard / detailed / constrained) and are selected by the
adaptive controller based on rolling performance metrics.
"""
from __future__ import annotations

import json
from typing import Optional

import anthropic
from rich.console import Console

from src.schemas import (
    AdaptiveConfig,
    NoveltyAssessment,
    PaperComponents,
    RelatedPaper,
    StructuredReview,
)
from src.retrieval import _build_schema

console = Console()

# ---------------------------------------------------------------------------
# Prompt templates (three variants)
# ---------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are an expert AI/ML researcher serving as a peer reviewer. "
    "You have broad knowledge of machine learning, deep learning, NLP, computer vision, "
    "and scientific methodology. Your reviews are thorough, fair, and constructive."
)

_VENUE_CRITERIA: dict[str, str] = {
    "NeurIPS": (
        "NeurIPS values: technical novelty, rigorous experiments, "
        "clarity of exposition, and broad impact on the ML community."
    ),
    "ICML": (
        "ICML values: theoretical or empirical novelty, strong experimental validation, "
        "and clear scientific contribution."
    ),
    "ICLR": (
        "ICLR values: representation learning advances, reproducibility, "
        "and open-science practices."
    ),
    "ACL": (
        "ACL values: linguistic insight, NLP methodology, evaluation rigour, "
        "and impact on language understanding/generation."
    ),
    "CVPR": (
        "CVPR values: computer vision novelty, strong benchmark results, "
        "and practical applicability."
    ),
    "Generic": (
        "Apply standard academic peer-review criteria: novelty, technical quality, "
        "experimental rigour, clarity, and significance."
    ),
}

_PROMPT_VARIANTS: dict[str, str] = {
    "standard": (
        "Generate a thorough and balanced peer review for the paper described below.\n"
        "Your review must:\n"
        "• Accurately reflect the paper's actual content\n"
        "• Cite specific sections, figures, tables, or equations when making claims\n"
        "• Be constructive — identify issues and suggest improvements\n"
        "• Only reference works that are explicitly mentioned in the 'Related papers' section\n"
        "• Avoid fabricating results, methods, or citations\n"
    ),
    "detailed": (
        "Generate a DETAILED peer review for the paper described below.\n"
        "Requirements (be especially thorough):\n"
        "• For every weakness, cite the specific section/experiment/claim that supports it\n"
        "• For every strength, explain WHY it matters to the field\n"
        "• Enumerate at least 3 concrete questions for the authors\n"
        "• Explicitly address: reproducibility, statistical significance, ablations, "
        "failure modes\n"
        "• Only reference works that are explicitly mentioned in the 'Related papers' section\n"
        "• Avoid fabricating results, methods, or citations\n"
    ),
    "constrained": (
        "Generate a peer review for the paper described below.\n"
        "STRICT CONSTRAINTS (follow exactly):\n"
        "• Every claim in your review must be directly traceable to the paper text\n"
        "• Do NOT infer or assume anything beyond what is explicitly stated in the paper\n"
        "• Do NOT mention papers, authors, or results not listed in 'Related papers'\n"
        "• When uncertain, ask a question rather than making an assertion\n"
        "• If the paper is unclear on a point, flag it as unclear rather than guessing\n"
    ),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_review(
    components: PaperComponents,
    related_papers: list[RelatedPaper],
    novelty: Optional[NoveltyAssessment],
    config: AdaptiveConfig,
    client: anthropic.Anthropic,
    venue: str = "NeurIPS",
    extra_criteria: str = "",
) -> StructuredReview:
    """
    Generate a structured review using Claude Opus 4.6 with adaptive thinking.

    Parameters
    ----------
    components   : Extracted paper components.
    related_papers: Retrieved related papers for context.
    novelty      : Novelty assessment (optional).
    config       : Adaptive runtime configuration.
    client       : Anthropic client.
    venue        : Target venue (NeurIPS, ICML, …).
    extra_criteria: Any additional reviewer criteria from the user.
    """
    console.print(
        f"[cyan]Generating {venue} review (prompt: {config.prompt_variant})…[/cyan]"
    )

    venue_criteria = _VENUE_CRITERIA.get(venue, _VENUE_CRITERIA["Generic"])
    prompt_instructions = _PROMPT_VARIANTS.get(
        config.prompt_variant, _PROMPT_VARIANTS["standard"]
    )

    # Build the context block
    related_str = _format_related_papers(related_papers)
    novelty_str = _format_novelty(novelty)

    user_content = (
        f"=== VENUE ===\n{venue}\n\n"
        f"=== VENUE CRITERIA ===\n{venue_criteria}\n\n"
        f"=== REVIEWER INSTRUCTIONS ===\n{prompt_instructions}\n"
        + (f"\nExtra criteria from user: {extra_criteria}\n" if extra_criteria else "")
        + f"\n=== PAPER COMPONENTS ===\n"
        f"Title: {components.title}\n"
        f"Authors: {', '.join(components.authors)}\n\n"
        f"Abstract:\n{components.abstract}\n\n"
        f"Problem Statement:\n{components.problem_statement}\n\n"
        f"Methodology:\n{components.methodology}\n\n"
        f"Experiments:\n{components.experiments}\n\n"
        f"Results:\n{components.results}\n\n"
        f"Main Claims:\n" + "\n".join(f"  • {c}" for c in components.main_claims) + "\n\n"
        f"Key Contributions:\n" + "\n".join(f"  • {c}" for c in components.key_contributions) + "\n\n"
        f"Baselines Mentioned:\n" + (", ".join(components.baselines_mentioned) or "None listed") + "\n\n"
        f"Datasets Used:\n" + (", ".join(components.datasets_used) or "None listed") + "\n\n"
        f"Limitations Acknowledged:\n" + "\n".join(f"  • {l}" for l in components.limitations_mentioned) + "\n\n"
        f"=== RELATED PAPERS (from Semantic Scholar) ===\n{related_str}\n\n"
        f"=== NOVELTY ASSESSMENT ===\n{novelty_str}\n\n"
        "Now generate the complete structured review in the required JSON format."
    )

    schema = _build_schema(StructuredReview)

    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": schema}},
        system=_SYSTEM_BASE,
        messages=[{"role": "user", "content": user_content}],
    )

    # The structured output is always in a text block (after any thinking blocks)
    text_block = next(b for b in response.content if b.type == "text")
    review = StructuredReview.model_validate_json(text_block.text)

    # Ensure the venue field is set correctly
    review = review.model_copy(update={"venue": venue, "paper_title": components.title})

    console.print(
        f"[green]✓ Review generated — Score: {review.score}/10 "
        f"({review.recommendation})[/green]"
    )

    # Optional second-pass critique
    if config.enable_second_pass:
        review = _second_pass_critique(review, components, client, config)

    return review


# ---------------------------------------------------------------------------
# Second-pass critique
# ---------------------------------------------------------------------------

def _second_pass_critique(
    review: StructuredReview,
    components: PaperComponents,
    client: anthropic.Anthropic,
    config: AdaptiveConfig,
) -> StructuredReview:
    """
    Run a second pass where Claude critiques its own review and produces
    a refined version.  Triggered when confidence is low or overall
    quality is uncertain.
    """
    console.print("[cyan]Running second-pass self-critique…[/cyan]")

    schema = _build_schema(StructuredReview)
    review_json = review.model_dump_json(indent=2)

    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": schema}},
        system=_SYSTEM_BASE,
        messages=[
            {
                "role": "user",
                "content": (
                    "Below is a draft peer review. Critically examine it:\n"
                    "1. Are all weaknesses grounded in actual paper content?\n"
                    "2. Are there any hallucinated claims or references?\n"
                    "3. Is the score well-justified?\n"
                    "4. Are there important aspects of the paper not covered?\n\n"
                    f"Paper title: {components.title}\n\n"
                    f"Draft review:\n{review_json}\n\n"
                    "Produce an improved version of the review that fixes any issues found."
                ),
            }
        ],
    )

    text_block = next(b for b in response.content if b.type == "text")
    refined = StructuredReview.model_validate_json(text_block.text)
    console.print("[green]✓ Second-pass critique complete[/green]")
    return refined


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_related_papers(papers: list[RelatedPaper]) -> str:
    if not papers:
        return "No related papers retrieved."
    lines = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        year = f" ({p.year})" if p.year else ""
        lines.append(
            f"{i}. {p.title}{year} — {authors}\n"
            f"   Relevance ({p.relevance_score:.2f}): {p.relevance_reason}"
        )
    return "\n".join(lines)


def _format_novelty(novelty: Optional[NoveltyAssessment]) -> str:
    if novelty is None:
        return "Novelty assessment not performed."
    return (
        f"Level: {novelty.novelty_level}\n"
        f"Novel aspects: {', '.join(novelty.novel_aspects) or 'none identified'}\n"
        f"Similar prior work: {', '.join(novelty.similar_prior_work) or 'none identified'}\n"
        f"Overlap concerns: {', '.join(novelty.overlap_concerns) or 'none'}\n"
        f"Justification: {novelty.novelty_justification}"
    )
