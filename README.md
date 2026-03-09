# 🔬 AI Research Paper Reviewer

An AI system that reads a research paper and writes a structured peer review — then grades its own work and gets better over time.

---

## What it does

```
You give it a PDF  →  It writes a review  →  It checks the review  →  It learns from mistakes
```

**Step by step:**

1. **Reads the paper** — extracts title, abstract, methods, experiments, claims, and contributions
2. **Finds related work** — searches Semantic Scholar for similar papers
3. **Checks novelty** — compares the paper's claims against prior work
4. **Writes a review** — generates a structured NeurIPS/ICML-style review with scores
5. **Grades itself** — a second AI pass checks the review for hallucinations, missing content, and unsupported claims
6. **Improves** — saves quality scores and automatically adjusts settings on the next run

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the web app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Web Interface

Upload a PDF → choose a venue → click **▶ Run Review**

| Tab | What you see |
|-----|-------------|
| 📄 Review | The full peer review with score, strengths, weaknesses, and questions |
| 🧑‍⚖️ Evaluation | Quality scores — was the review grounded? did it hallucinate? |
| 📚 Related Papers | Papers found on Semantic Scholar + novelty assessment |
| 📊 Metrics Dashboard | History of all past runs and performance trends |

---

## CLI (optional)

```bash
# Basic review
python main.py paper.pdf

# Specify venue
python main.py paper.pdf --venue ICML

# Add custom focus
python main.py paper.pdf --criteria "Focus on fairness and reproducibility"

# Save full output as JSON
python main.py paper.pdf --output review.json

# View performance history
python main.py --stats
```

---

## How it self-improves

After each review, quality scores are saved locally. Before the next run, the system reads those scores and adjusts automatically:

| Problem detected | What changes |
|-----------------|-------------|
| Review makes things up | Switches to a stricter prompt |
| Review misses paper content | Fetches more related papers, uses a more thorough prompt |
| Review quality is low | Runs a second self-critique pass |

---

## Supported venues

`NeurIPS` · `ICML` · `ICLR` · `ACL` · `CVPR` · `Generic`

---

## Project structure

```
├── app.py              ← Streamlit web interface
├── main.py             ← CLI interface
├── src/
│   ├── extractor.py    ← Reads PDF, extracts paper structure
│   ├── retrieval.py    ← Searches Semantic Scholar, checks novelty
│   ├── reviewer.py     ← Generates the review
│   ├── evaluator.py    ← Grades the review quality
│   ├── monitor.py      ← Saves and displays metrics history
│   ├── adaptive.py     ← Adjusts settings based on past performance
│   └── schemas.py      ← Data models
├── data/metrics/       ← Review history (auto-created)
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- Anthropic API key ([get one here](https://console.anthropic.com))
- Optional: Semantic Scholar API key for higher rate limits
