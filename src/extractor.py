"""
Paper text extraction and structured component extraction.

Supports PDF (via PyMuPDF) and plain text files.
Uses Claude to identify and extract key paper components.
"""
from __future__ import annotations

import re
from pathlib import Path

import anthropic
from rich.console import Console

from src.schemas import PaperComponents

console = Console()

# Keep well under the 200K context window; reserve space for the prompt
MAX_PAPER_CHARS = 80_000


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. "
            "Install with: pip install pymupdf"
        )

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def extract_text_from_file(file_path: str) -> str:
    """Extract raw text from a PDF or plain-text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        console.print(f"[dim]Extracting text from PDF: {path.name}[/dim]")
        return extract_text_from_pdf(file_path)

    # Assume UTF-8 text (also handles .txt, .md, .tex, etc.)
    console.print(f"[dim]Reading text file: {path.name}[/dim]")
    return path.read_text(encoding="utf-8", errors="replace")


def _truncate_for_context(text: str, max_chars: int = MAX_PAPER_CHARS) -> str:
    """
    Truncate paper text to fit within the context window.
    Keeps the beginning (abstract/intro) and end (conclusion/references),
    dropping the middle if needed.
    """
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    truncation_notice = (
        "\n\n[... MIDDLE SECTION TRUNCATED TO FIT CONTEXT WINDOW ...]\n\n"
    )
    console.print(
        f"[yellow]Paper text truncated from {len(text):,} to ~{max_chars:,} chars[/yellow]"
    )
    return text[:half] + truncation_notice + text[-half:]


def _clean_text(text: str) -> str:
    """Basic cleanup: collapse excessive whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_paper_components(
    text: str,
    client: anthropic.Anthropic,
) -> PaperComponents:
    """
    Use Claude to extract structured components from paper text.

    Returns a validated PaperComponents object.
    """
    console.print("[cyan]Extracting paper components...[/cyan]")

    cleaned = _clean_text(_truncate_for_context(text))

    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=(
            "You are an expert at analyzing AI/ML research papers. "
            "Extract the requested components accurately from the paper text provided. "
            "If a field is not present or applicable, use an empty list or a brief note."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze the following research paper and extract its key components.\n\n"
                    "---PAPER TEXT START---\n"
                    f"{cleaned}\n"
                    "---PAPER TEXT END---\n\n"
                    "Extract:\n"
                    "- title and authors\n"
                    "- abstract (verbatim or very close paraphrase)\n"
                    "- problem statement\n"
                    "- methodology (technical approach)\n"
                    "- experiments (what was tested, how)\n"
                    "- results (key numbers/findings)\n"
                    "- main claims (explicit statements the paper makes)\n"
                    "- baselines mentioned or compared against\n"
                    "- datasets used\n"
                    "- key contributions (as stated by the authors)\n"
                    "- limitations acknowledged by the authors"
                ),
            }
        ],
        output_format=PaperComponents,
    )

    components: PaperComponents = response.parsed_output
    console.print(
        f"[green]✓ Extracted components for:[/green] {components.title[:80]}"
    )
    return components
