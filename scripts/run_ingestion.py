"""
DevBrain Ingestion Pipeline — Main Entry Point

This script orchestrates the full data pipeline:
  1. Load raw docs and issues from data/raw/ (already downloaded by loaders)
  2. Chunk them with source-appropriate strategies
  3. Embed all chunks via OpenAI API
  4. Upsert everything into Pinecone

Run with:
  uv run python -m scripts.run_ingestion

Prerequisites:
  - API keys set in .env (OpenAI, Pinecone)
  - Raw data downloaded (run the loaders first if data/raw/ is empty):
      uv run python -m src.ingestion.github_docs_loader
      uv run python -m src.ingestion.github_issues_loader

Cost estimate:
  ~5,000 chunks × ~250 tokens/chunk = ~1.25M tokens
  At $0.02/1M tokens = ~$0.025 (less than 3 cents)
"""

import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Add project root to path ─────────────────────────────────────
# When running as `python -m scripts.run_ingestion`, Python needs
# to find the `src` package. This ensures it works regardless of
# how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DOCS_DIR,
    RAW_ISSUES_DIR,
    NAMESPACE_DOCS,
    NAMESPACE_ISSUES,
)
from src.chunking.markdown_chunker import chunk_documents, print_chunk_stats
from src.chunking.issue_chunker import chunk_issues
from src.embeddings.openai_embedder import embed_chunks, estimate_cost
from src.vectorstore.pinecone_store import (
    get_or_create_index,
    upsert_chunks,
    get_index_stats,
)

console = Console()


# ── Step 1: Load raw data from disk ──────────────────────────────

def load_raw_docs() -> list[dict]:
    """
    Load previously downloaded markdown docs from data/raw/docs/.

    Why load from disk instead of re-downloading?
    - Faster (no GitHub API calls)
    - Doesn't consume rate limit
    - Reproducible (same input every run)
    - Separates "fetching" from "processing"

    Each .md file becomes a document dict with content + metadata.
    We skip _manifest.json and any non-.md files.
    """
    documents = []
    docs_path = RAW_DOCS_DIR

    if not docs_path.exists() or not any(docs_path.rglob("*.md")):
        console.print(
            "[red]❌ No docs found in data/raw/docs/[/red]\n"
            "   Run the docs loader first:\n"
            "   uv run python -m src.ingestion.github_docs_loader"
        )
        return []

    # Walk all .md files recursively
    for md_file in sorted(docs_path.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")

        # Skip very short files
        if len(content.strip()) < 50:
            continue

        # Build relative path for metadata
        relative_path = md_file.relative_to(docs_path)
        title = md_file.stem  # filename without .md

        documents.append({
            "content": content,
            "metadata": {
                "source": "docs",
                "file_path": str(relative_path),
                "title": title,
                "url": f"https://github.com/fastapi/fastapi/blob/master/docs/en/docs/{relative_path}",
            },
        })

    console.print(f"  Loaded [cyan]{len(documents)}[/cyan] doc files from disk")
    return documents


def load_raw_issues() -> list[dict]:
    """
    Load previously downloaded issues from data/raw/issues/.

    Each issue is stored as a JSON file: issue_12345.json
    We read each one and reconstruct the document dict.
    """
    documents = []
    issues_path = RAW_ISSUES_DIR

    if not issues_path.exists() or not any(issues_path.glob("issue_*.json")):
        console.print(
            "[red]❌ No issues found in data/raw/issues/[/red]\n"
            "   Run the issues loader first:\n"
            "   uv run python -m src.ingestion.github_issues_loader"
        )
        return []

    for json_file in sorted(issues_path.glob("issue_*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))

        # Skip if content is too short
        if len(data.get("content", "").strip()) < 50:
            continue

        documents.append(data)

    console.print(f"  Loaded [cyan]{len(documents)}[/cyan] issues from disk")
    return documents


# ── Main pipeline ─────────────────────────────────────────────────

def run_pipeline():
    """
    Execute the full ingestion pipeline.

    The pipeline has 4 stages, each clearly separated:
      Stage 1: Load raw data from disk
      Stage 2: Chunk documents
      Stage 3: Embed chunks (this calls the OpenAI API — costs money)
      Stage 4: Upsert to Pinecone

    Between stages 2 and 3, we show a cost estimate and ask for
    confirmation. This prevents accidental spending if you
    accidentally run the pipeline with 100,000 documents.
    """
    start_time = time.time()

    console.print(Panel(
        "[bold]DevBrain Ingestion Pipeline[/bold]\n"
        "Load → Chunk → Embed → Store",
        style="blue",
    ))

    # ── Stage 1: Load ─────────────────────────────────────────────
    console.print("\n[bold]📥 Stage 1: Loading raw data from disk[/bold]")

    docs = load_raw_docs()
    issues = load_raw_issues()

    if not docs and not issues:
        console.print("[red]No data to process. Run the loaders first.[/red]")
        return

    # ── Stage 2: Chunk ────────────────────────────────────────────
    console.print("\n[bold]✂️  Stage 2: Chunking documents[/bold]")

    console.print("\n  [bold]Docs chunking:[/bold]")
    doc_chunks = chunk_documents(docs) if docs else []
    if doc_chunks:
        print_chunk_stats(doc_chunks)

    console.print("\n  [bold]Issues chunking:[/bold]")
    issue_chunks = chunk_issues(issues) if issues else []

    all_chunks = doc_chunks + issue_chunks
    console.print(f"\n  Total chunks: [cyan]{len(all_chunks)}[/cyan]")

    # ── Cost estimate + confirmation ──────────────────────────────
    console.print("\n[bold]💰 Cost Estimate[/bold]")

    all_texts = [c["content"] for c in all_chunks]
    cost = estimate_cost(all_texts)

    cost_table = Table()
    cost_table.add_column("Metric", style="bold")
    cost_table.add_column("Value", style="cyan")
    cost_table.add_row("Total chunks", str(cost["num_texts"]))
    cost_table.add_row("Total tokens", f"{cost['total_tokens']:,}")
    cost_table.add_row("Avg tokens/chunk", str(cost["avg_tokens_per_text"]))
    cost_table.add_row("Estimated cost", f"${cost['estimated_cost_usd']:.4f}")
    console.print(cost_table)

    # Ask for confirmation before spending money
    console.print("\n  [yellow]This will call the OpenAI API (costs real money).[/yellow]")
    confirm = input("  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        console.print("[red]Aborted.[/red]")
        return

    # ── Stage 3: Embed ────────────────────────────────────────────
    console.print("\n[bold]🔢 Stage 3: Embedding chunks[/bold]")

    console.print("\n  [bold]Embedding doc chunks:[/bold]")
    doc_chunks = embed_chunks(doc_chunks) if doc_chunks else []

    console.print("\n  [bold]Embedding issue chunks:[/bold]")
    issue_chunks = embed_chunks(issue_chunks) if issue_chunks else []

    # ── Stage 4: Store ────────────────────────────────────────────
    console.print("\n[bold]🌲 Stage 4: Storing in Pinecone[/bold]")

    index = get_or_create_index()

    if doc_chunks:
        console.print(f"\n  Upserting {len(doc_chunks)} doc chunks...")
        upsert_chunks(index, doc_chunks, namespace=NAMESPACE_DOCS)

    if issue_chunks:
        console.print(f"\n  Upserting {len(issue_chunks)} issue chunks...")
        upsert_chunks(index, issue_chunks, namespace=NAMESPACE_ISSUES)

    # ── Summary ───────────────────────────────────────────────────
    # Wait a moment for Pinecone to index the vectors
    console.print("\n  Waiting for Pinecone to index vectors...")
    time.sleep(5)

    stats = get_index_stats(index)
    elapsed = time.time() - start_time

    console.print(Panel(
        f"[bold green]✅ Pipeline Complete![/bold green]\n\n"
        f"  Total vectors: {stats['total_vectors']}\n"
        f"  Namespaces:\n"
        + "\n".join(
            f"    • {ns}: {count} vectors"
            for ns, count in stats["namespaces"].items()
        )
        + f"\n\n  Time: {elapsed:.1f} seconds"
        + f"\n  Cost: ~${cost['estimated_cost_usd']:.4f}",
        style="green",
    ))

    console.print(
        "\n[bold]Next step:[/bold] Verify with:\n"
        "  uv run python -m scripts.verify_pinecone"
    )


if __name__ == "__main__":
    run_pipeline()