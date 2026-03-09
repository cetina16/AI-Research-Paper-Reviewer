"""
Streamlit web interface for the AI Research Paper Reviewer.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import streamlit as st
from dotenv import load_dotenv

from src.adaptive import AdaptiveController
from src.evaluator import evaluate_review
from src.extractor import extract_paper_components, extract_text_from_file
from src.monitor import MetricsStore
from src.retrieval import assess_novelty, get_related_papers
from src.reviewer import generate_review
from src.schemas import ReviewMetrics

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Paper Reviewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_VENUES = ["NeurIPS", "ICML", "ICLR", "ACL", "CVPR", "Generic"]

_SCORE_COLOURS = {
    range(1, 4): "🔴",
    range(4, 6): "🟠",
    range(6, 8): "🟡",
    range(8, 11): "🟢",
}


def _score_emoji(score: int) -> str:
    for r, emoji in _SCORE_COLOURS.items():
        if score in r:
            return emoji
    return "⚪"


def _metric_colour(val: float, invert: bool = False) -> str:
    v = (1 - val) if invert else val
    if v >= 0.7:
        return "normal"
    if v >= 0.4:
        return "off"
    return "inverse"


# ── Session-state keys ───────────────────────────────────────────────────────
_KEYS = [
    "review", "evaluation", "components",
    "related_papers", "novelty", "paper_text",
    "metrics_saved", "review_id",
]


def _clear_results() -> None:
    for k in _KEYS:
        st.session_state.pop(k, None)


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    st.sidebar.title("🔬 Paper Reviewer")
    st.sidebar.markdown("Upload a research paper and generate a structured peer review.")
    st.sidebar.divider()

    uploaded = st.sidebar.file_uploader(
        "Upload paper (PDF or TXT)",
        type=["pdf", "txt", "md", "tex"],
        help="PDF is recommended. Plain text files are also supported.",
        on_change=_clear_results,
    )

    venue = st.sidebar.selectbox("Target venue", _VENUES, index=0)

    criteria = st.sidebar.text_area(
        "Extra reviewer criteria (optional)",
        placeholder="e.g. Focus on fairness, reproducibility, and evaluation methodology.",
        height=80,
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Pipeline options**")
    skip_retrieval = st.sidebar.checkbox(
        "Skip literature retrieval",
        help="Faster, but the review won't cite related work.",
    )
    skip_eval = st.sidebar.checkbox(
        "Skip quality evaluation",
        help="Don't run the LLM-as-judge step after review generation.",
    )

    st.sidebar.divider()
    run = st.sidebar.button(
        "▶ Run Review",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
    )

    return dict(
        uploaded=uploaded,
        venue=venue,
        criteria=criteria,
        skip_retrieval=skip_retrieval,
        skip_eval=skip_eval,
        run=run,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(opts: dict) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not set.**\n\n"
            "Add it to your `.env` file or set it as an environment variable."
        )
        return

    client = anthropic.Anthropic(api_key=api_key)
    store = MetricsStore()
    controller = AdaptiveController(store)
    review_id = str(uuid.uuid4())
    start = time.time()

    uploaded = opts["uploaded"]
    suffix = Path(uploaded.name).suffix or ".pdf"

    with st.status("Running review pipeline…", expanded=True) as status:

        # ── Save upload to temp file ─────────────────────────────────────────
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            # ── Adaptive config ──────────────────────────────────────────────
            st.write("⚙️ Loading adaptive configuration…")
            config = controller.get_config()

            # ── Extract text ─────────────────────────────────────────────────
            st.write("📄 Extracting text from paper…")
            paper_text = extract_text_from_file(tmp_path)

            # ── Extract components ────────────────────────────────────────────
            st.write("🔍 Extracting paper components…")
            components = extract_paper_components(paper_text, client)

            # ── Retrieval ─────────────────────────────────────────────────────
            related_papers = []
            novelty = None
            if not opts["skip_retrieval"]:
                st.write("🌐 Searching related literature (Semantic Scholar)…")
                related_papers = get_related_papers(components, client, config)

                st.write("💡 Assessing novelty…")
                novelty = assess_novelty(components, related_papers, client)

            # ── Generate review ───────────────────────────────────────────────
            st.write("✍️ Generating structured review…")
            review = generate_review(
                components=components,
                related_papers=related_papers,
                novelty=novelty,
                config=config,
                client=client,
                venue=opts["venue"],
                extra_criteria=opts["criteria"],
            )

            # ── Evaluate ──────────────────────────────────────────────────────
            evaluation = None
            if not opts["skip_eval"]:
                st.write("🧑‍⚖️ Evaluating review quality…")
                evaluation = evaluate_review(
                    review=review,
                    components=components,
                    paper_text=paper_text,
                    related_papers=related_papers,
                    client=client,
                )

            elapsed = time.time() - start

            # ── Save metrics ──────────────────────────────────────────────────
            if evaluation:
                metrics = ReviewMetrics(
                    review_id=review_id,
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    paper_title=components.title,
                    venue=opts["venue"],
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
                st.session_state["metrics_saved"] = True

            # ── Persist to session state ──────────────────────────────────────
            st.session_state["review"] = review
            st.session_state["evaluation"] = evaluation
            st.session_state["components"] = components
            st.session_state["related_papers"] = related_papers
            st.session_state["novelty"] = novelty
            st.session_state["paper_text"] = paper_text
            st.session_state["review_id"] = review_id

            status.update(
                label=f"✅ Done in {elapsed:.1f}s — Score: {review.score}/10 ({review.recommendation})",
                state="complete",
            )

        except Exception as exc:
            status.update(label="❌ Pipeline failed", state="error")
            st.error(f"**Error:** {exc}")
            raise
        finally:
            os.unlink(tmp_path)


# ── Tab: Review ───────────────────────────────────────────────────────────────
def tab_review() -> None:
    review = st.session_state.get("review")
    if review is None:
        st.info("Upload a paper and click **▶ Run Review** to get started.")
        return

    emoji = _score_emoji(review.score)
    rec_colour = (
        "green" if "Accept" in review.recommendation
        else ("orange" if "Borderline" in review.recommendation or "Weak" in review.recommendation
              else "red")
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"## {review.venue} Review")
    st.markdown(f"### {review.paper_title}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{emoji} {review.score} / 10")
    c2.metric("Confidence", f"{review.confidence} / 5")
    c3.markdown(
        f"**Recommendation**  \n"
        f"<span style='color:{rec_colour};font-size:1.1em;font-weight:600'>"
        f"{review.recommendation}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Summary ───────────────────────────────────────────────────────────────
    st.subheader("Summary")
    st.markdown(review.summary)

    # ── Strengths / Weaknesses side by side ───────────────────────────────────
    col_s, col_w = st.columns(2)
    with col_s:
        st.subheader("✅ Strengths")
        for s in review.strengths:
            st.markdown(f"- {s}")
    with col_w:
        st.subheader("⚠️ Weaknesses")
        for w in review.weaknesses:
            st.markdown(f"- {w}")

    # ── Questions ─────────────────────────────────────────────────────────────
    st.subheader("❓ Questions for Authors")
    for i, q in enumerate(review.questions, 1):
        st.markdown(f"{i}. {q}")

    # ── Limitations ───────────────────────────────────────────────────────────
    st.subheader("🔒 Limitations")
    for l in review.limitations:
        st.markdown(f"- {l}")

    # ── Technical deep-dives ──────────────────────────────────────────────────
    col_n, col_t = st.columns(2)
    with col_n:
        st.subheader("🆕 Novelty")
        st.markdown(review.novelty_assessment)
    with col_t:
        st.subheader("⚙️ Technical Quality")
        st.markdown(review.technical_quality)

    # ── Optional sections ─────────────────────────────────────────────────────
    if review.related_works_missing:
        with st.expander("📚 Missing Related Works"):
            for w in review.related_works_missing:
                st.markdown(f"- {w}")

    if review.unsupported_claims:
        with st.expander("🚩 Unsupported Claims"):
            for c in review.unsupported_claims:
                st.markdown(f"- {c}")

    if review.ethical_concerns and review.ethical_concerns.lower() != "none identified.":
        with st.expander("⚖️ Ethical Concerns"):
            st.markdown(review.ethical_concerns)

    st.divider()
    st.subheader("📝 Justification")
    st.markdown(review.justification)

    # ── Adaptive config info ──────────────────────────────────────────────────
    components = st.session_state.get("components")
    if components:
        with st.expander("ℹ️ Run details"):
            st.markdown(f"**Review ID:** `{st.session_state.get('review_id', 'N/A')}`")
            st.markdown(f"**Authors:** {', '.join(components.authors)}")
            st.markdown(f"**Datasets:** {', '.join(components.datasets_used) or 'none listed'}")
            st.markdown(f"**Baselines:** {', '.join(components.baselines_mentioned) or 'none listed'}")


# ── Tab: Evaluation ───────────────────────────────────────────────────────────
def tab_evaluation() -> None:
    evaluation = st.session_state.get("evaluation")
    if evaluation is None:
        if st.session_state.get("review"):
            st.info("Quality evaluation was skipped. Uncheck **Skip quality evaluation** and re-run.")
        else:
            st.info("Run a review first to see evaluation results.")
        return

    st.subheader("Review Quality Evaluation")
    st.caption("Scores from an independent LLM-as-judge pass on the generated review.")

    # ── Metric grid ───────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    def _pct(v: float) -> str:
        return f"{v:.0%}"

    m1.metric("Groundedness", _pct(evaluation.groundedness_score),
              help="Criticisms grounded in paper content")
    m2.metric("Reference Validity", _pct(evaluation.reference_validity_score),
              help="Cited related works are real and relevant")
    m3.metric("Completeness", _pct(evaluation.completeness_score),
              help="Review covers all expected sections")
    m4.metric("Structure Compliance", _pct(evaluation.structure_compliance_score),
              help="Matches the target venue review template")
    m5.metric("Hallucination Rate", _pct(evaluation.hallucination_rate),
              help="Lower is better — fraction of fabricated content",
              delta=f"{-evaluation.hallucination_rate:.0%}" if evaluation.hallucination_rate > 0 else None,
              delta_color="inverse")
    m6.metric("Overall Quality", _pct(evaluation.overall_quality_score),
              help="Weighted combination of all dimensions")

    st.divider()

    # ── Progress bars ─────────────────────────────────────────────────────────
    st.markdown("**Dimension breakdown**")
    dims = [
        ("Groundedness", evaluation.groundedness_score, False),
        ("Reference Validity", evaluation.reference_validity_score, False),
        ("Completeness", evaluation.completeness_score, False),
        ("Structure Compliance", evaluation.structure_compliance_score, False),
        ("No Hallucination", 1 - evaluation.hallucination_rate, False),
    ]
    for label, val, _ in dims:
        colour = "green" if val >= 0.7 else ("orange" if val >= 0.4 else "red")
        st.markdown(f"**{label}** — {val:.0%}")
        st.progress(val)

    st.divider()

    # ── Issues ────────────────────────────────────────────────────────────────
    issue_sections = [
        ("🔗 Groundedness Issues", evaluation.groundedness_issues),
        ("🌀 Hallucination Examples", evaluation.hallucination_examples),
        ("📖 Invalid References", evaluation.invalid_references),
        ("📋 Missing Sections", evaluation.missing_sections),
    ]
    for title, items in issue_sections:
        if items:
            with st.expander(f"{title} ({len(items)})"):
                for item in items:
                    st.markdown(f"- {item}")

    # ── Suggestions ───────────────────────────────────────────────────────────
    if evaluation.improvement_suggestions:
        st.subheader("💡 Improvement Suggestions")
        for s in evaluation.improvement_suggestions:
            st.markdown(f"- {s}")


# ── Tab: Related Papers ───────────────────────────────────────────────────────
def tab_related_papers() -> None:
    related = st.session_state.get("related_papers", [])
    novelty = st.session_state.get("novelty")

    if not st.session_state.get("review"):
        st.info("Run a review first.")
        return

    if not related:
        st.info("No related papers retrieved. Uncheck **Skip literature retrieval** and re-run.")
        return

    # ── Novelty summary ───────────────────────────────────────────────────────
    if novelty:
        colours = {"high": "green", "medium": "blue", "low": "orange", "incremental": "red"}
        colour = colours.get(novelty.novelty_level.lower(), "gray")
        st.markdown(
            f"**Novelty level:** "
            f"<span style='color:{colour};font-weight:700;font-size:1.1em'>"
            f"{novelty.novelty_level.upper()}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(novelty.novelty_justification)

        col_a, col_b = st.columns(2)
        with col_a:
            if novelty.novel_aspects:
                st.markdown("**Novel aspects**")
                for a in novelty.novel_aspects:
                    st.markdown(f"- {a}")
        with col_b:
            if novelty.overlap_concerns:
                st.markdown("**Overlap concerns**")
                for c in novelty.overlap_concerns:
                    st.markdown(f"- {c}")

        st.divider()

    # ── Papers table ──────────────────────────────────────────────────────────
    st.subheader(f"Retrieved Papers ({len(related)})")

    import pandas as pd
    rows = []
    for p in related:
        authors_str = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        rows.append({
            "Title": p.title,
            "Authors": authors_str,
            "Year": p.year or "—",
            "Citations": p.citation_count or 0,
            "Relevance": f"{p.relevance_score:.2f}",
            "Reason": p.relevance_reason,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Tab: Metrics Dashboard ────────────────────────────────────────────────────
def tab_metrics() -> None:
    store = MetricsStore()
    stats = store.aggregate_stats(n=20)
    records = store.recent(n=20)

    st.subheader("Performance Dashboard")
    st.caption("Rolling averages over the last 20 review runs stored locally.")

    if not stats:
        st.info("No metrics yet. Run your first review to start tracking performance.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total reviews", int(stats["count"]))
    k2.metric("Avg groundedness", f"{stats['avg_groundedness']:.0%}")
    k3.metric("Avg hallucination", f"{stats['avg_hallucination_rate']:.0%}",
              delta_color="inverse")
    k4.metric("Avg overall quality", f"{stats['avg_overall_quality']:.0%}")
    k5.metric("Avg processing time", f"{stats['avg_processing_time']:.0f}s")

    st.divider()

    # ── Recent runs table ─────────────────────────────────────────────────────
    st.subheader("Recent Runs")
    if records:
        import pandas as pd
        rows = [
            {
                "ID": r.review_id[:8] + "…",
                "Paper": r.paper_title[:45] + ("…" if len(r.paper_title) > 45 else ""),
                "Venue": r.venue,
                "Score": f"{r.review_score}/10",
                "Quality": f"{r.overall_quality_score:.0%}",
                "Hallucination": f"{r.hallucination_rate:.0%}",
                "Prompt": r.prompt_variant,
                "2nd pass": "✓" if r.second_pass_used else "—",
                "Time": f"{r.processing_time_seconds:.0f}s",
            }
            for r in reversed(records)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Adaptive adjustments hint ─────────────────────────────────────────────
    if stats["avg_groundedness"] < 0.60 or stats["avg_hallucination_rate"] > 0.25:
        st.warning(
            "The adaptive controller will adjust settings on the next run "
            "to address low groundedness or high hallucination rates."
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    opts = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔬 AI Research Paper Reviewer")
    st.markdown(
        "Powered by **Claude Opus 4.6** · RAG via Semantic Scholar · "
        "Self-improving via LLM-as-judge feedback loop"
    )
    st.divider()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if opts["run"] and opts["uploaded"]:
        _clear_results()
        run_pipeline(opts)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 Review", "🧑‍⚖️ Evaluation", "📚 Related Papers", "📊 Metrics Dashboard"]
    )
    with tab1:
        tab_review()
    with tab2:
        tab_evaluation()
    with tab3:
        tab_related_papers()
    with tab4:
        tab_metrics()


if __name__ == "__main__":
    main()
