"""
Verify that the Pinecone index has data and returns sensible results.

This script:
  1. Connects to Pinecone and shows index stats
  2. Embeds a few sample questions
  3. Queries each namespace and shows the top results with scores
  4. Helps you verify that retrieval is working before Phase 2

Run with:
  uv run python -m scripts.verify_pinecone

What to look for:
  - Scores above 0.7 = good match (the result is relevant to the question)
  - Scores 0.5-0.7 = okay match (somewhat related)
  - Scores below 0.5 = poor match (probably not relevant)
  - If ALL scores are below 0.5, something went wrong with chunking or embedding
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import NAMESPACE_DOCS, NAMESPACE_ISSUES
from src.embeddings.openai_embedder import embed_texts
from src.vectorstore.pinecone_store import get_or_create_index, get_index_stats, query_index

console = Console()


# ── Test questions ────────────────────────────────────────────────
# These are designed to test different retrieval scenarios:
TEST_QUESTIONS = [
    {
        "question": "How do I add CORS middleware to FastAPI?",
        "namespace": NAMESPACE_DOCS,
        "why": "Basic how-to → should match docs about CORS",
    },
    {
        "question": "How does dependency injection work in FastAPI?",
        "namespace": NAMESPACE_DOCS,
        "why": "Core concept → should match tutorial docs about Depends()",
    },
    {
        "question": "422 Unprocessable Entity error with Pydantic model",
        "namespace": NAMESPACE_ISSUES,
        "why": "Error message → should match issues reporting this error",
    },
    {
        "question": "How to use WebSocket in FastAPI?",
        "namespace": NAMESPACE_DOCS,
        "why": "Feature question → should match WebSocket documentation",
    },
    {
        "question": "background tasks not working",
        "namespace": NAMESPACE_ISSUES,
        "why": "Bug-like query → should match issues about BackgroundTasks",
    },
]


def run_verification():
    """
    Run sample queries against Pinecone and display results.

    For each test question:
    1. Embed the question using OpenAI (same model used for chunks)
    2. Query Pinecone in the specified namespace
    3. Display top 3 results with scores and snippet preview

    This is your sanity check — if the results make sense,
    the pipeline worked correctly and you're ready for Phase 2.
    """
    console.print(Panel(
        "[bold]DevBrain — Pinecone Verification[/bold]\n"
        "Testing retrieval with sample questions",
        style="blue",
    ))

    # ── Index stats ───────────────────────────────────────────────
    index = get_or_create_index()
    stats = get_index_stats(index)

    stats_table = Table(title="📊 Index Statistics")
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", style="cyan")
    stats_table.add_row("Total vectors", str(stats["total_vectors"]))
    for ns, count in stats["namespaces"].items():
        stats_table.add_row(f"Namespace: {ns}", f"{count} vectors")
    console.print(stats_table)

    if stats["total_vectors"] == 0:
        console.print(
            "\n[red]❌ Index is empty! Run the ingestion pipeline first:[/red]\n"
            "   uv run python -m scripts.run_ingestion"
        )
        return

    # ── Embed all test questions in one batch ─────────────────────
    console.print("\n[bold]Embedding test questions...[/bold]")
    questions = [q["question"] for q in TEST_QUESTIONS]
    question_embeddings = embed_texts(questions)

    # ── Query each question ───────────────────────────────────────
    for i, test in enumerate(TEST_QUESTIONS):
        console.print(f"\n{'─' * 70}")
        console.print(
            f"\n[bold]Question {i+1}:[/bold] {test['question']}"
            f"\n[dim]Searching namespace: {test['namespace']} | {test['why']}[/dim]\n"
        )

        results = query_index(
            index=index,
            query_embedding=question_embeddings[i],
            namespace=test["namespace"],
            top_k=3,
        )

        if not results:
            console.print("[yellow]  No results found[/yellow]")
            continue

        for rank, result in enumerate(results, 1):
            # Color the score based on quality
            score = result["score"]
            if score >= 0.7:
                score_color = "green"
                quality = "GOOD"
            elif score >= 0.5:
                score_color = "yellow"
                quality = "OK"
            else:
                score_color = "red"
                quality = "POOR"

            # Truncate text for display
            preview = result["text"][:200].replace("\n", " ")
            if len(result["text"]) > 200:
                preview += "..."

            # Get source info
            title = result["metadata"].get("title", "unknown")
            url = result["metadata"].get("url", "")

            console.print(
                f"  [bold]#{rank}[/bold] "
                f"[{score_color}]Score: {score:.4f} ({quality})[/{score_color}]"
            )
            console.print(f"      Title: {title}")
            console.print(f"      URL: {url}")
            console.print(f"      Preview: [dim]{preview}[/dim]")
            console.print()

    # ── Summary ───────────────────────────────────────────────────
    console.print("─" * 70)
    console.print(Panel(
        "[bold green]✅ Verification Complete[/bold green]\n\n"
        "What to look for:\n"
        "  • Scores > 0.7 = retrieval is working well\n"
        "  • Results match the question topic = chunking is good\n"
        "  • Metadata (title, url) is correct = citations will work\n\n"
        "If results look good, you're ready for Phase 2!\n"
        "Phase 2 will connect a retrieval chain + LLM to answer questions.",
        style="green",
    ))


if __name__ == "__main__":
    run_verification()