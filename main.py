#!/usr/bin/env python3
"""
AI Research Paper Reviewer — CLI entry point.

Usage
─────
  python main.py paper.pdf
  python main.py paper.pdf --venue ICML
  python main.py paper.pdf --venue NeurIPS --criteria "Focus on fairness"
  python main.py --stats          # show historical performance
  python main.py --history        # show recent review runs

The pipeline (per run)
──────────────────────
  1. Load AdaptiveConfig (derived from rolling metrics)
  2. Extract text from PDF / text file
  3. Extract structured PaperComponents   (Claude)
  4. Retrieve related papers              (Semantic Scholar)
  5. Assess novelty                       (Claude + adaptive thinking)
  6. Generate structured review           (Claude + adaptive thinking)
  7. Evaluate review quality              (Claude LLM-as-judge)
  8. Persist metrics
  9. Display results
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from src.adaptive import AdaptiveController
from src.evaluator import evaluate_review
from src.extractor import extract_paper_components, extract_text_from_file
from src.monitor import MetricsStore
from src.retrieval import assess_novelty, get_related_papers
from src.reviewer import generate_review
from src.schemas import ReviewMetrics

# ---------------------------------------------------------------------------
load_dotenv()
console = Console()

_SUPPORTED_VENUES = ["NeurIPS", "ICML", "ICLR", "ACL", "CVPR", "Generic"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-reviewer",
        description="Self-Improving AI Research Paper Reviewer",
    )
    p.add_argument(
        "paper",
        nargs="?",
        help="Path to the PDF or text file to review",
    )
    p.add_argument(
        "--venue",
        default="NeurIPS",
        choices=_SUPPORTED_VENUES,
        help="Target review venue (default: NeurIPS)",
    )
    p.add_argument(
        "--criteria",
        default="",
        help="Additional reviewer criteria or focus areas (freeform text)",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Show aggregate performance statistics and exit",
    )
    p.add_argument(
        "--history",
        action="store_true",
        help="Show recent review run history and exit",
    )
    p.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip Semantic Scholar retrieval (faster, less context)",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the quality evaluation step",
    )
    p.add_argument(
        "--output",
        default="",
        help="Path to write the review JSON (optional)",
    )
    return p


# ---------------------------------------------------------------------------
# Review pipeline
# ---------------------------------------------------------------------------

def run_review(args: argparse.Namespace) -> None:
    # ── Setup ────────────────────────────────────────────────────────────────
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[red]Error: ANTHROPIC_API_KEY environment variable not set.[/red]\n"
            "Set it in .env or export it in your shell."
        )
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    store = MetricsStore()
    controller = AdaptiveController(store)

    review_id = str(uuid.uuid4())
    start_time = time.time()

    console.print(Rule("[bold cyan]AI Research Paper Reviewer[/bold cyan]"))
    console.print(f"[dim]Review ID: {review_id}[/dim]")
    console.print(f"[dim]Venue: {args.venue} | Adaptive thinking: enabled[/dim]")
    console.print()

    # ── Step 1: Adaptive config ───────────────────────────────────────────────
    config = controller.get_config()
    console.print(
        f"[dim]Config → model={config.model}, "
        f"retrieval_depth={config.retrieval_depth}, "
        f"prompt={config.prompt_variant}, "
        f"second_pass={config.enable_second_pass}[/dim]"
    )
    console.print()

    # ── Step 2: Extract text ─────────────────────────────────────────────────
    console.print(Rule("Step 1 · Text Extraction"))
    paper_text = extract_text_from_file(args.paper)
    console.print(f"[dim]Extracted {len(paper_text):,} characters[/dim]")
    console.print()

    # ── Step 3: Extract paper components ────────────────────────────────────
    console.print(Rule("Step 2 · Paper Component Extraction"))
    components = extract_paper_components(paper_text, client)
    console.print()

    # ── Step 4: Retrieve related papers ─────────────────────────────────────
    related_papers = []
    novelty = None
    if not args.no_retrieval:
        console.print(Rule("Step 3 · Literature Retrieval"))
        related_papers = get_related_papers(components, client, config)
        console.print()

        # ── Step 5: Novelty assessment ───────────────────────────────────────
        console.print(Rule("Step 4 · Novelty Assessment"))
        novelty = assess_novelty(components, related_papers, client)
        console.print()
    else:
        console.print("[dim]Skipping retrieval & novelty assessment (--no-retrieval)[/dim]")
        console.print()

    # ── Step 6: Generate review ──────────────────────────────────────────────
    console.print(Rule("Step 5 · Review Generation"))
    review = generate_review(
        components=components,
        related_papers=related_papers,
        novelty=novelty,
        config=config,
        client=client,
        venue=args.venue,
        extra_criteria=args.criteria,
    )
    console.print()

    # ── Step 7: Quality evaluation ────────────────────────────────────────────
    evaluation = None
    if not args.no_eval:
        console.print(Rule("Step 6 · Review Quality Evaluation"))
        evaluation = evaluate_review(
            review=review,
            components=components,
            paper_text=paper_text,
            related_papers=related_papers,
            client=client,
        )
        console.print()

    elapsed = time.time() - start_time

    # ── Step 8: Persist metrics ───────────────────────────────────────────────
    if evaluation:
        metrics = ReviewMetrics(
            review_id=review_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            paper_title=components.title,
            venue=args.venue,
            model_used=config.model,
            retrieval_depth=len(related_papers),
            prompt_variant=config.prompt_variant,
            processing_time_seconds=round(elapsed, 2),
            groundedness_score=evaluation.groundedness_score,
            hallucination_rate=evaluation.hallucination_rate,
            reference_validity_score=evaluation.reference_validity_score,
            completeness_score=evaluation.completeness_score,
            structure_compliance_score=evaluation.structure_compliance_score,
            overall_quality_score=evaluation.overall_quality_score,
            review_score=review.score,
            review_confidence=review.confidence,
            second_pass_used=config.enable_second_pass,
        )
        store.save(metrics)

    # ── Step 9: Display results ───────────────────────────────────────────────
    console.print(Rule("[bold green]Review Output[/bold green]"))
    _display_review(review)

    if evaluation:
        console.print()
        console.print(Rule("[bold]Quality Evaluation[/bold]"))
        _display_evaluation(evaluation)

    console.print()
    console.print(
        f"[dim]Total time: {elapsed:.1f}s | "
        f"Related papers: {len(related_papers)} | "
        f"Prompt variant: {config.prompt_variant}[/dim]"
    )

    # ── Optional JSON output ─────────────────────────────────────────────────
    if args.output:
        import json
        output = {
            "review_id": review_id,
            "review": review.model_dump(),
            "evaluation": evaluation.model_dump() if evaluation else None,
            "config": config.model_dump(),
            "components": {
                "title": components.title,
                "authors": components.authors,
                "main_claims": components.main_claims,
                "key_contributions": components.key_contributions,
            },
            "related_papers": [p.model_dump() for p in related_papers],
            "novelty": novelty.model_dump() if novelty else None,
            "processing_time_seconds": round(elapsed, 2),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        console.print(f"[dim]Full output saved to: {args.output}[/dim]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_review(review) -> None:
    md_lines = [
        f"# {review.venue} Review: {review.paper_title}",
        "",
        "## Summary",
        review.summary,
        "",
        "## Strengths",
        *[f"- {s}" for s in review.strengths],
        "",
        "## Weaknesses",
        *[f"- {w}" for w in review.weaknesses],
        "",
        "## Questions for Authors",
        *[f"{i+1}. {q}" for i, q in enumerate(review.questions)],
        "",
        "## Limitations",
        *[f"- {l}" for l in review.limitations],
        "",
        "## Novelty Assessment",
        review.novelty_assessment,
        "",
        "## Technical Quality",
        review.technical_quality,
    ]

    if review.related_works_missing:
        md_lines += ["", "## Missing Related Works"]
        md_lines += [f"- {w}" for w in review.related_works_missing]

    if review.unsupported_claims:
        md_lines += ["", "## Unsupported Claims"]
        md_lines += [f"- {c}" for c in review.unsupported_claims]

    if review.ethical_concerns and review.ethical_concerns.lower() != "none identified.":
        md_lines += ["", "## Ethical Concerns", review.ethical_concerns]

    md_lines += [
        "",
        "---",
        f"**Score:** {review.score}/10 | "
        f"**Confidence:** {review.confidence}/5 | "
        f"**Recommendation:** {review.recommendation}",
        "",
        "### Justification",
        review.justification,
    ]

    console.print(Markdown("\n".join(md_lines)))


def _display_evaluation(evaluation) -> None:
    console.print(
        Panel(
            f"[cyan]Groundedness:[/cyan]        {evaluation.groundedness_score:.2f}\n"
            f"[cyan]Hallucination rate:[/cyan]  {evaluation.hallucination_rate:.2f}  (lower is better)\n"
            f"[cyan]Reference validity:[/cyan]  {evaluation.reference_validity_score:.2f}\n"
            f"[cyan]Completeness:[/cyan]        {evaluation.completeness_score:.2f}\n"
            f"[cyan]Structure compliance:[/cyan]{evaluation.structure_compliance_score:.2f}\n"
            f"[bold green]Overall quality:[/bold green]     {evaluation.overall_quality_score:.2f}",
            title="Evaluation Scores",
            expand=False,
        )
    )
    if evaluation.improvement_suggestions:
        console.print("\n[yellow]Suggestions for improvement:[/yellow]")
        for s in evaluation.improvement_suggestions:
            console.print(f"  • {s}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    store = MetricsStore()

    if args.stats:
        console.print(Rule("[bold cyan]Performance Statistics[/bold cyan]"))
        store.print_stats()
        return

    if args.history:
        console.print(Rule("[bold cyan]Recent Review History[/bold cyan]"))
        store.print_recent()
        return

    if not args.paper:
        parser.print_help()
        sys.exit(1)

    run_review(args)


if __name__ == "__main__":
    main()
