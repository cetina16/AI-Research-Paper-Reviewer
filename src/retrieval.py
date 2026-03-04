"""
Related literature retrieval via Semantic Scholar and novelty assessment.

Semantic Scholar Graph API (free tier, no auth required):
  https://api.semanticscholar.org/graph/v1/paper/search
Rate limit: ~100 requests / 5 minutes without an API key.
Set SEMANTIC_SCHOLAR_API_KEY in .env for higher limits.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
import anthropic
from rich.console import Console

from src.schemas import (
    AdaptiveConfig,
    NoveltyAssessment,
    PaperComponents,
    RelatedPaper,
    RelatedPapersRanking,
)

console = Console()

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,authors,year,abstract,citationCount"


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

def _s2_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Query Semantic Scholar paper search endpoint.
    Returns a list of raw paper dicts (may be empty on failure).
    """
    headers: dict[str, str] = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    params = {"query": query, "limit": limit, "fields": _S2_FIELDS}

    try:
        with httpx.Client(timeout=20.0) as http:
            resp = http.get(f"{_S2_BASE}/paper/search", params=params, headers=headers)
            resp.raise_for_status()
            return resp.json().get("data", [])
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            console.print("[yellow]Semantic Scholar rate limit hit, waiting 10 s…[/yellow]")
            time.sleep(10)
        else:
            console.print(
                f"[yellow]Semantic Scholar HTTP {exc.response.status_code}: {exc}[/yellow]"
            )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]Semantic Scholar search failed: {exc}[/yellow]")

    return []


def _raw_to_snippet(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": raw.get("title", ""),
        "authors": [a.get("name", "") for a in raw.get("authors", [])],
        "year": raw.get("year"),
        "abstract": (raw.get("abstract") or "")[:300],
        "citation_count": raw.get("citationCount", 0),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_related_papers(
    components: PaperComponents,
    client: anthropic.Anthropic,
    config: AdaptiveConfig,
) -> list[RelatedPaper]:
    """
    1. Search Semantic Scholar with the paper title and key-claim keywords.
    2. Use Claude to rank and assess the relevance of candidates.
    Returns up to config.retrieval_depth RelatedPaper objects.
    """
    depth = config.retrieval_depth
    console.print(f"[cyan]Retrieving up to {depth} related papers from Semantic Scholar…[/cyan]")

    # ── Search ──────────────────────────────────────────────────────────────
    title_hits = _s2_search(components.title, limit=depth + 5)

    # Build keyword query from first two claims + problem statement snippet
    kw_parts = components.main_claims[:2] + [components.problem_statement[:80]]
    kw_query = " ".join(kw_parts)[:200]
    keyword_hits = _s2_search(kw_query, limit=depth + 5)

    # Deduplicate (exclude the paper under review itself)
    seen: set[str] = {components.title.lower()}
    candidates: list[dict[str, Any]] = []
    for raw in title_hits + keyword_hits:
        t = (raw.get("title") or "").lower()
        if t and t not in seen:
            seen.add(t)
            candidates.append(raw)

    if not candidates:
        console.print("[yellow]No related papers found via Semantic Scholar.[/yellow]")
        return []

    # ── Rank with Claude ─────────────────────────────────────────────────────
    snippets = json.dumps(
        [_raw_to_snippet(p) for p in candidates[:20]], indent=2, ensure_ascii=False
    )

    ranking_response = client.messages.parse(
        model="claude-haiku-4-5",  # lightweight model for ranking task
        max_tokens=3072,
        system=(
            "You are an expert at assessing the relevance of research papers. "
            "Select and rank the most relevant papers from the provided list."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f'Paper under review: "{components.title}"\n\n'
                    f"Key contributions:\n"
                    + "\n".join(f"  • {c}" for c in components.key_contributions)
                    + f"\n\nMain claims:\n"
                    + "\n".join(f"  • {c}" for c in components.main_claims[:5])
                    + f"\n\nCandidate related papers (from Semantic Scholar):\n{snippets}\n\n"
                    f"Select the top {depth} most relevant papers. "
                    "Exclude any paper that is clearly the same paper under review. "
                    "For each selected paper, provide a relevance_score (0–1) and "
                    "a one-sentence relevance_reason."
                ),
            }
        ],
        output_format=RelatedPapersRanking,
    )

    papers = ranking_response.parsed_output.papers[:depth]
    console.print(f"[green]✓ Retrieved {len(papers)} related papers[/green]")
    return papers


def assess_novelty(
    components: PaperComponents,
    related_papers: list[RelatedPaper],
    client: anthropic.Anthropic,
) -> NoveltyAssessment:
    """
    Use Claude with adaptive thinking to assess how novel the paper is
    relative to retrieved related work.
    """
    console.print("[cyan]Assessing novelty…[/cyan]")

    if related_papers:
        related_str = "\n".join(
            f"- [{p.year or '?'}] {p.title} (cited {p.citation_count or 0}×): {p.relevance_reason}"
            for p in related_papers
        )
    else:
        related_str = "No related papers were retrieved."

    # Build the JSON schema from the Pydantic model
    schema = _build_schema(NoveltyAssessment)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": schema}},
        system=(
            "You are an expert AI researcher assessing the novelty of a paper "
            "relative to prior work. Be rigorous and honest."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f'Paper: "{components.title}"\n\n'
                    f"Key contributions:\n"
                    + "\n".join(f"  • {c}" for c in components.key_contributions)
                    + f"\n\nMain claims:\n"
                    + "\n".join(f"  • {c}" for c in components.main_claims)
                    + f"\n\nRelated prior work retrieved:\n{related_str}\n\n"
                    "Assess how novel this paper is. Identify what is genuinely new, "
                    "what overlaps with prior work, and any concerns about insufficient novelty."
                ),
            }
        ],
    )

    text_block = next(b for b in response.content if b.type == "text")
    novelty = NoveltyAssessment.model_validate_json(text_block.text)
    console.print(f"[green]✓ Novelty level: {novelty.novelty_level}[/green]")
    return novelty


# ---------------------------------------------------------------------------
# Schema helper
# ---------------------------------------------------------------------------

def _build_schema(model_class: type) -> dict:
    """
    Build a JSON schema dict from a Pydantic model, ensuring every
    'object' node has additionalProperties: false (required by Claude
    structured outputs).
    """
    schema = model_class.model_json_schema()
    _enforce_no_additional(schema)
    # Process $defs if present
    for def_schema in schema.get("$defs", {}).values():
        _enforce_no_additional(def_schema)
    return schema


def _enforce_no_additional(schema: dict) -> None:
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for prop in schema.get("properties", {}).values():
            _enforce_no_additional(prop)
    elif schema.get("type") == "array" and "items" in schema:
        _enforce_no_additional(schema["items"])
    for sub in schema.get("anyOf", []):
        _enforce_no_additional(sub)
