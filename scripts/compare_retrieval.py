"""
Compare retrieval strategies side-by-side.

This script runs the same questions through 4 retrieval strategies
and shows you the results so you can see the improvement:

  1. Dense only (Pinecone semantic search)
  2. Sparse only (BM25 keyword search)
  3. Hybrid (Dense + Sparse + RRF, no reranking)
  4. Hybrid + Reranking (the full pipeline)

Run with:
  uv run python -m scripts.compare_retrieval

This is the PROOF that advanced retrieval is worth the complexity.
If scores don't improve, something is wrong.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever

console = Console()


# ── Test questions designed to show different strengths ────────────
TEST_QUESTIONS = [
    {
        "question": "How do I add CORS middleware to FastAPI?",
        "type": "how-to",
        "why": "Semantic query — dense should do well, hybrid should be similar",
    },
    {
        "question": "422 Unprocessable Entity error",
        "type": "error",
        "why": "Keyword-heavy — BM25 should find exact error code matches",
    },
    {
        "question": "BackgroundTasks not executing after response",
        "type": "bug",
        "why": "Mixed — needs both semantic (concept) and keyword (BackgroundTasks)",
    },
    {
        "question": "OAuth2PasswordBearer token authentication",
        "type": "specific-api",
        "why": "Specific class name — BM25 finds exact matches, dense finds related docs",
    },
    {
        "question": "How to make my API faster with async",
        "type": "conceptual",
        "why": "Conceptual — dense excels here, BM25 might miss 'performance' docs",
    },
]


def run_comparison():
    """Run all test questions through all retrieval strategies."""
    console.print(Panel(
        "[bold]🔬 Retrieval Strategy Comparison[/bold]\n"
        "Same questions, 4 different retrieval methods.\n"
        "Watch how each method handles different query types.",
        style="blue",
    ))

    # Initialize (builds BM25 index — takes a few seconds)
    retriever = HybridRetriever(top_k=3, use_reranker=True)

    for q_idx, test in enumerate(TEST_QUESTIONS, 1):
        question = test["question"]
        console.print(f"\n{'═' * 75}")
        console.print(
            f"\n[bold]Question {q_idx}/{len(TEST_QUESTIONS)}:[/bold] {question}"
            f"\n[dim]Type: {test['type']} | {test['why']}[/dim]\n"
        )

        # Run all 4 strategies
        strategies = {
            "Dense Only (Pinecone)": retriever.retrieve_dense_only(question),
            "Sparse Only (BM25)": retriever.retrieve_sparse_only(question),
            "Hybrid (no rerank)": retriever.retrieve_without_reranking(question),
            "Hybrid + Reranking": retriever.retrieve(question),
        }

        # Build comparison table
        table = Table(title=f"Top 3 Results Comparison", show_lines=True)
        table.add_column("Strategy", style="bold", width=22)
        table.add_column("#1 Result", width=25)
        table.add_column("#2 Result", width=25)
        table.add_column("#3 Result", width=25)

        for strategy_name, results in strategies.items():
            cells = []
            for r in results[:3]:
                title = r["metadata"].get("title", "?")[:20]
                source = r.get("namespace", "?")[:5]

                # Pick the most meaningful score to display
                if "rerank_score" in r:
                    score_str = f"rr={r['rerank_score']:.3f}"
                elif "rrf_score" in r:
                    score_str = f"rrf={r['rrf_score']:.4f}"
                else:
                    score_str = f"s={r['score']:.3f}"

                cells.append(f"[{source}] {title}\n{score_str}")

            # Pad if less than 3 results
            while len(cells) < 3:
                cells.append("[dim]no result[/dim]")

            table.add_row(strategy_name, *cells)

        console.print(table)

    # Summary
    console.print(f"\n{'═' * 75}")
    console.print(Panel(
        "[bold green]What to look for:[/bold green]\n\n"
        "• [bold]Error/keyword queries[/bold] (like '422 error'):\n"
        "  BM25 and Hybrid should find exact matches that Dense misses.\n\n"
        "• [bold]Conceptual queries[/bold] (like 'make API faster'):\n"
        "  Dense should find related docs even without exact keywords.\n\n"
        "• [bold]Hybrid + Reranking[/bold] should consistently have\n"
        "  the most relevant result in position #1.\n\n"
        "• If Hybrid + Reranking doesn't improve over Dense-only,\n"
        "  check your Cohere API key and BM25 index size.",
        style="green",
    ))


if __name__ == "__main__":
    run_comparison()