"""
Adaptive improvement controller.

Reads rolling performance metrics from the MetricsStore and returns
an AdaptiveConfig that adjusts the review pipeline's behaviour.

Decision rules
──────────────
Low groundedness (< 0.60)
    → increase retrieval depth (more related-paper context)
    → switch to 'detailed' prompt (forces citing specific sections)

High hallucination rate (> 0.25)
    → switch to 'constrained' prompt (strictest grounding rules)

Low reference validity (< 0.55)
    → increase retrieval depth (better related-paper pool)

Low completeness (< 0.65)
    → switch to 'detailed' prompt (forces thorough coverage)

Low overall quality (< 0.50)
    → enable second-pass critique

Multiple issues simultaneously
    → combine the adjustments above, cap retrieval depth at 15

High cost signal (avg processing time > 120 s)
    → keep current depth but note the latency concern

No metrics yet (first run)
    → use sensible defaults
"""
from __future__ import annotations

from rich.console import Console

from src.monitor import MetricsStore
from src.schemas import AdaptiveConfig

console = Console()

# ── Thresholds ───────────────────────────────────────────────────────────────
_GROUNDEDNESS_LOW = 0.60
_HALLUCINATION_HIGH = 0.25
_REFERENCE_VALIDITY_LOW = 0.55
_COMPLETENESS_LOW = 0.65
_OVERALL_QUALITY_LOW = 0.50
_PROCESSING_TIME_HIGH = 120.0  # seconds

_BASE_RETRIEVAL_DEPTH = 5
_INCREASED_RETRIEVAL_DEPTH = 8
_MAX_RETRIEVAL_DEPTH = 15


class AdaptiveController:
    """
    Derives a new AdaptiveConfig from rolling metrics.
    Each call is stateless — it reads from the MetricsStore every time.
    """

    def __init__(self, store: MetricsStore) -> None:
        self.store = store

    def get_config(self) -> AdaptiveConfig:
        """
        Compute and return the next AdaptiveConfig.
        Logs all decisions to the console.
        """
        stats = self.store.aggregate_stats(n=20)

        if not stats:
            console.print(
                "[dim]No prior metrics found — using default configuration.[/dim]"
            )
            return AdaptiveConfig()

        console.print(
            f"[dim]Adaptive controller: analysing {int(stats['count'])} prior review(s)…[/dim]"
        )

        adjustments: list[str] = []
        retrieval_depth = _BASE_RETRIEVAL_DEPTH
        prompt_variant = "standard"
        enable_second_pass = False

        # ── Rule 1: Low groundedness ─────────────────────────────────────────
        if stats["avg_groundedness"] < _GROUNDEDNESS_LOW:
            retrieval_depth = max(retrieval_depth, _INCREASED_RETRIEVAL_DEPTH)
            prompt_variant = "detailed"
            adjustments.append(
                f"groundedness={stats['avg_groundedness']:.2f} < {_GROUNDEDNESS_LOW} "
                "→ increased retrieval depth, switched to 'detailed' prompt"
            )

        # ── Rule 2: High hallucination rate ─────────────────────────────────
        if stats["avg_hallucination_rate"] > _HALLUCINATION_HIGH:
            prompt_variant = "constrained"
            adjustments.append(
                f"hallucination_rate={stats['avg_hallucination_rate']:.2f} > {_HALLUCINATION_HIGH} "
                "→ switched to 'constrained' prompt"
            )

        # ── Rule 3: Low reference validity ──────────────────────────────────
        if stats["avg_reference_validity"] < _REFERENCE_VALIDITY_LOW:
            retrieval_depth = max(retrieval_depth, _INCREASED_RETRIEVAL_DEPTH)
            adjustments.append(
                f"reference_validity={stats['avg_reference_validity']:.2f} < {_REFERENCE_VALIDITY_LOW} "
                "→ increased retrieval depth"
            )

        # ── Rule 4: Low completeness ─────────────────────────────────────────
        if stats["avg_completeness"] < _COMPLETENESS_LOW:
            if prompt_variant != "constrained":
                prompt_variant = "detailed"
            adjustments.append(
                f"completeness={stats['avg_completeness']:.2f} < {_COMPLETENESS_LOW} "
                "→ switched to 'detailed' prompt"
            )

        # ── Rule 5: Low overall quality ──────────────────────────────────────
        if stats["avg_overall_quality"] < _OVERALL_QUALITY_LOW:
            enable_second_pass = True
            adjustments.append(
                f"overall_quality={stats['avg_overall_quality']:.2f} < {_OVERALL_QUALITY_LOW} "
                "→ enabled second-pass critique"
            )

        # ── Rule 6: Cap retrieval depth ──────────────────────────────────────
        retrieval_depth = min(retrieval_depth, _MAX_RETRIEVAL_DEPTH)

        # ── Rule 7: Latency note ─────────────────────────────────────────────
        if stats["avg_processing_time"] > _PROCESSING_TIME_HIGH:
            adjustments.append(
                f"avg_processing_time={stats['avg_processing_time']:.0f}s > {_PROCESSING_TIME_HIGH}s "
                "— note: high latency detected (consider async batch for bulk reviews)"
            )

        if adjustments:
            console.print("[yellow]Adaptive adjustments applied:[/yellow]")
            for adj in adjustments:
                console.print(f"  [yellow]• {adj}[/yellow]")
        else:
            console.print("[green]Performance looks good — keeping default settings.[/green]")

        return AdaptiveConfig(
            model="claude-opus-4-6",
            retrieval_depth=retrieval_depth,
            prompt_variant=prompt_variant,
            enable_second_pass=enable_second_pass,
            adjustments_made=adjustments,
        )
