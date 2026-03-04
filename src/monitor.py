"""
Performance monitoring and metrics persistence.

Metrics are stored as newline-delimited JSON in data/metrics/metrics.jsonl.
Each line is one ReviewMetrics record.  The MetricsStore class provides
helpers to save, load, and aggregate these records.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from src.schemas import ReviewMetrics

console = Console()

_DEFAULT_METRICS_PATH = Path(__file__).parent.parent / "data" / "metrics" / "metrics.jsonl"


class MetricsStore:
    """Persistent store for ReviewMetrics records."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _DEFAULT_METRICS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ────────────────────────────────────────────────────────────────

    def save(self, metrics: ReviewMetrics) -> None:
        """Append a single ReviewMetrics record to the store."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(metrics.model_dump_json() + "\n")
        console.print(
            f"[dim]Metrics saved → {self.path.name} (id={metrics.review_id[:8]}…)[/dim]"
        )

    # ── Read ─────────────────────────────────────────────────────────────────

    def load_all(self) -> list[ReviewMetrics]:
        """Load all stored ReviewMetrics records."""
        if not self.path.exists():
            return []
        records: list[ReviewMetrics] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(ReviewMetrics.model_validate_json(line))
                    except Exception:  # noqa: BLE001
                        pass  # skip malformed lines
        return records

    def recent(self, n: int = 10) -> list[ReviewMetrics]:
        """Return the n most recent records."""
        return self.load_all()[-n:]

    # ── Aggregate stats ──────────────────────────────────────────────────────

    def aggregate_stats(self, n: int = 20) -> dict[str, float]:
        """
        Compute rolling averages over the last n records.
        Returns a dict with metric names → float averages.
        Returns an empty dict if no records exist.
        """
        records = self.recent(n)
        if not records:
            return {}

        def _avg(values: list[float]) -> float:
            return statistics.mean(values) if values else 0.0

        return {
            "count": len(records),
            "avg_groundedness": _avg([r.groundedness_score for r in records]),
            "avg_hallucination_rate": _avg([r.hallucination_rate for r in records]),
            "avg_reference_validity": _avg([r.reference_validity_score for r in records]),
            "avg_completeness": _avg([r.completeness_score for r in records]),
            "avg_structure_compliance": _avg([r.structure_compliance_score for r in records]),
            "avg_overall_quality": _avg([r.overall_quality_score for r in records]),
            "avg_processing_time": _avg([r.processing_time_seconds for r in records]),
            "avg_score": _avg([float(r.review_score) for r in records]),
            "second_pass_rate": sum(1 for r in records if r.second_pass_used) / len(records),
        }

    # ── Display ───────────────────────────────────────────────────────────────

    def print_stats(self, n: int = 20) -> None:
        """Print a Rich table of aggregate stats."""
        stats = self.aggregate_stats(n)
        if not stats:
            console.print("[yellow]No metrics stored yet.[/yellow]")
            return

        table = Table(title=f"Rolling Performance Metrics (last {int(stats['count'])} reviews)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        def _status(val: float, good: float = 0.7, bad: float = 0.4, invert: bool = False) -> str:
            if invert:
                val = 1.0 - val
            if val >= good:
                return "[green]✓[/green]"
            elif val >= bad:
                return "[yellow]~[/yellow]"
            return "[red]✗[/red]"

        rows = [
            ("Groundedness", stats["avg_groundedness"], _status(stats["avg_groundedness"])),
            ("Hallucination rate", stats["avg_hallucination_rate"], _status(stats["avg_hallucination_rate"], good=0.1, bad=0.3, invert=True)),
            ("Reference validity", stats["avg_reference_validity"], _status(stats["avg_reference_validity"])),
            ("Completeness", stats["avg_completeness"], _status(stats["avg_completeness"])),
            ("Structure compliance", stats["avg_structure_compliance"], _status(stats["avg_structure_compliance"])),
            ("Overall quality", stats["avg_overall_quality"], _status(stats["avg_overall_quality"])),
            ("Avg review score", stats["avg_score"] / 10.0, _status(stats["avg_score"] / 10.0)),
            ("Avg processing time", stats["avg_processing_time"], ""),
            ("Second-pass rate", stats["second_pass_rate"], ""),
        ]
        for name, val, status in rows:
            if name in ("Avg processing time",):
                table.add_row(name, f"{val:.1f}s", status)
            elif name in ("Second-pass rate",):
                table.add_row(name, f"{val:.0%}", status)
            else:
                table.add_row(name, f"{val:.3f}", status)

        console.print(table)

    def print_recent(self, n: int = 5) -> None:
        """Print a table of the most recent review runs."""
        records = self.recent(n)
        if not records:
            console.print("[yellow]No review history yet.[/yellow]")
            return

        table = Table(title=f"Last {len(records)} Review Runs")
        table.add_column("ID", style="dim")
        table.add_column("Paper (truncated)")
        table.add_column("Venue")
        table.add_column("Score", justify="right")
        table.add_column("Quality", justify="right")
        table.add_column("Prompt")

        for r in reversed(records):
            table.add_row(
                r.review_id[:8] + "…",
                r.paper_title[:40] + ("…" if len(r.paper_title) > 40 else ""),
                r.venue,
                f"{r.review_score}/10",
                f"{r.overall_quality_score:.2f}",
                r.prompt_variant,
            )
        console.print(table)
