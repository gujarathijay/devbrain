"""
Conversation-aware chunker for GitHub Issues.

Why a separate chunker for issues?
Issues are structurally different from documentation:

1. Many issues are SHORT (under 1000 chars) — these should stay as
   one chunk. Splitting a 400-char issue into two 200-char chunks
   destroys context and creates two useless fragments.

2. Some issues are LONG (5000+ chars) — detailed bug reports with
   stack traces, reproduction steps, and multiple code blocks.
   These need splitting, but at natural boundaries (paragraphs),
   not mid-sentence.

3. Issues don't have markdown headers (## / ###) — they're more
   like a conversation. Paragraph breaks (\n\n) are the natural
   split point.

Strategy:
- If issue content < chunk_size: keep as ONE chunk (don't split)
- If issue content >= chunk_size: split on paragraph boundaries
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


# ── Issue-specific separators ────────────────────────────────────
# Issues don't have markdown headers, so our priority is:
# 1. Paragraph breaks (most natural for conversational text)
# 2. Newlines
# 3. Spaces (emergency)
ISSUE_SEPARATORS = [
    "\n\n",       # paragraph boundary (most common in issues)
    "\n",         # line break
    " ",          # word boundary (fallback)
]


def create_issue_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter configured for GitHub issues.

    Same chunk_size/overlap as docs, but different separators
    because issues don't have markdown headers.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=ISSUE_SEPARATORS,
        strip_whitespace=True,
        keep_separator=True,
    )


def chunk_issues(documents: list[dict]) -> list[dict]:
    """
    Split issue documents into chunks, keeping short issues whole.

    The key decision here:

      if len(content) <= CHUNK_SIZE:
          → Keep as ONE chunk. Don't split.

      else:
          → Split using paragraph-based recursive splitter.

    Why keep short issues whole?
    A short issue like:
      "Issue #9876: Cannot use Optional[int] as query param
       Status: closed | Labels: bug

       When I define `q: Optional[int] = None`, FastAPI raises
       a validation error. Fixed by using `Union[int, None]`."

    This is 200 chars. If we split it, we'd get two useless fragments.
    As one chunk, it's a perfectly self-contained Q&A pair.

    Returns list of chunk dicts with inherited metadata.
    """
    splitter = create_issue_splitter()
    all_chunks = []

    short_kept_whole = 0
    long_split = 0

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        if len(content) <= CHUNK_SIZE:
            # Short issue — keep as one chunk
            chunk_metadata = {
                **metadata,
                "chunk_index": 0,
                "total_chunks": 1,
            }
            all_chunks.append({
                "content": content,
                "metadata": chunk_metadata,
            })
            short_kept_whole += 1

        else:
            # Long issue — split it
            text_chunks = splitter.split_text(content)

            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                }
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata,
                })
            long_split += 1

    from rich.console import Console
    console = Console()
    console.print(
        f"  Issues kept whole: [cyan]{short_kept_whole}[/cyan] | "
        f"Issues split: [cyan]{long_split}[/cyan] | "
        f"Total chunks: [cyan]{len(all_chunks)}[/cyan]"
    )

    return all_chunks


def print_chunk_stats(chunks: list[dict]) -> None:
    """Print statistics about issue chunking results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    sizes = [len(c["content"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0

    # Count by state (open vs closed)
    states: dict[str, int] = {}
    for chunk in chunks:
        state = chunk["metadata"].get("state", "unknown")
        states[state] = states.get(state, 0) + 1

    table = Table(title="📊 Issue Chunking Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Total chunks", str(len(chunks)))
    table.add_row("Avg chunk size", f"{avg_size:.0f} chars")
    table.add_row("Min chunk size", f"{min(sizes)} chars")
    table.add_row("Max chunk size", f"{max(sizes)} chars")
    for state, count in states.items():
        table.add_row(f"State: {state}", str(count))

    console.print(table)


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: uv run python -m src.chunking.issue_chunker

    Tests with two sample issues: one short (stays whole), one long (gets split).
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Sample 1: SHORT issue (should stay as one chunk)
    short_issue = {
        "content": (
            "Issue #9876: Cannot use Optional[int] as query parameter\n"
            "Status: closed | Labels: bug, answered | Created: 2024-01-15\n\n"
            "When I define `q: Optional[int] = None` as a query parameter,\n"
            "FastAPI raises a validation error on missing values.\n\n"
            "Solution: Use `Union[int, None] = None` instead of Optional[int].\n"
            "This was fixed in FastAPI 0.100.0."
        ),
        "metadata": {
            "source": "issues",
            "issue_number": 9876,
            "title": "Cannot use Optional[int] as query parameter",
            "url": "https://github.com/fastapi/fastapi/issues/9876",
            "state": "closed",
            "labels": ["bug", "answered"],
        },
    }

    # Sample 2: LONG issue (should be split)
    long_body = "\n\n".join([
        f"Paragraph {i}: This is a detailed paragraph about a complex bug "
        f"involving authentication middleware, dependency injection, and "
        f"WebSocket connections. It includes stack traces, configuration "
        f"details, and reproduction steps that make the issue quite lengthy."
        for i in range(15)
    ])
    long_issue = {
        "content": (
            "Issue #5432: Complex auth middleware breaks WebSocket deps\n"
            "Status: open | Labels: bug, help wanted | Created: 2024-06-01\n\n"
            f"{long_body}"
        ),
        "metadata": {
            "source": "issues",
            "issue_number": 5432,
            "title": "Complex auth middleware breaks WebSocket deps",
            "url": "https://github.com/fastapi/fastapi/issues/5432",
            "state": "open",
            "labels": ["bug", "help wanted"],
        },
    }

    console.print("\n[bold]✂️  Testing Issue Chunker[/bold]\n")

    console.print(f"[bold]Short issue:[/bold] {len(short_issue['content'])} chars")
    console.print(f"[bold]Long issue:[/bold] {len(long_issue['content'])} chars")
    console.print(f"[bold]Chunk size threshold:[/bold] {CHUNK_SIZE} chars\n")

    chunks = chunk_issues([short_issue, long_issue])

    console.print()
    for i, chunk in enumerate(chunks):
        label = f"Issue #{chunk['metadata']['issue_number']}"
        console.print(Panel(
            chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""),
            title=f"Chunk {i} — {label} ({len(chunk['content'])} chars)",
            subtitle=f"chunk_index={chunk['metadata']['chunk_index']} "
                     f"of {chunk['metadata']['total_chunks']}",
        ))

    print_chunk_stats(chunks)