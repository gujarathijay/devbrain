"""
Markdown-aware chunker for documentation files.

Why a specialized chunker for markdown?
Regular text splitters don't understand markdown structure.
They might split inside a code block, between a header and its content,
or cut a table in half. This chunker uses markdown-specific separators
so splits happen at natural section boundaries.

The separator priority order:
  1. "\n## "   → H2 headers (major topic changes)
  2. "\n### "  → H3 headers (subtopic changes)
  3. "\n#### " → H4 headers (sub-subtopics)
  4. "\n\n"    → Blank lines (paragraph boundaries)
  5. "\n"      → Single newlines (line breaks)
  6. " "       → Spaces (word boundaries, last resort)

LangChain's RecursiveCharacterTextSplitter tries the FIRST separator.
If all resulting pieces are under chunk_size, it's done. If any piece
is still too large, it tries the NEXT separator on that piece, and so on.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


# ── Markdown separators (priority order) ─────────────────────────
# These are the boundaries where we PREFER to split.
# The splitter tries them in order: biggest boundary first.
MARKDOWN_SEPARATORS = [
    "\n## ",      # H2 — major sections like "## Authentication"
    "\n### ",     # H3 — subsections like "### OAuth2 with Password"
    "\n#### ",    # H4 — sub-subsections
    "\n\n",       # blank line — paragraph boundary
    "\n",         # single newline — line break
    " ",          # space — word boundary (emergency fallback)
]


def create_markdown_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter configured for markdown documents.

    Parameters explained:
    - chunk_size (1000): target size in characters per chunk.
      Why 1000? It's roughly 200-250 tokens, which is:
      • Big enough to contain a complete thought/example
      • Small enough that the embedding captures specific meaning
      • Fits comfortably in an LLM context window (even 20 chunks = ~5K tokens)

    - chunk_overlap (200): characters shared between adjacent chunks.
      Why 200? About 1-2 sentences of overlap. Enough to preserve
      context at boundaries without too much redundancy.

    - separators: our markdown-specific list (defined above)

    - strip_whitespace (True): removes leading/trailing whitespace
      from each chunk. Clean chunks = better embeddings.

    - keep_separator (True): includes the separator in the chunk.
      Without this, "### OAuth2 with Password\n\nTo use OAuth2..."
      would lose the header and become just "To use OAuth2..." —
      losing critical context about what section this belongs to.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=MARKDOWN_SEPARATORS,
        strip_whitespace=True,
        keep_separator=True,
    )


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split a list of documents into chunks, preserving metadata.

    Input: list of document dicts from the docs loader
      [
        {
          "content": "# First Steps\n\nYou can create a FastAPI app...",
          "metadata": {"source": "docs", "title": "first-steps", "url": "..."}
        },
        ...
      ]

    Output: list of chunk dicts, each with inherited metadata + chunk_index
      [
        {
          "content": "# First Steps\n\nYou can create a FastAPI app...",
          "metadata": {"source": "docs", "title": "first-steps", "url": "...", "chunk_index": 0}
        },
        {
          "content": "### Run it\n\nRun the server with...",
          "metadata": {"source": "docs", "title": "first-steps", "url": "...", "chunk_index": 1}
        },
        ...
      ]
    """
    splitter = create_markdown_splitter()
    all_chunks = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        # Split this document's content into chunks
        # split_text returns a list of strings
        text_chunks = splitter.split_text(content)

        for i, chunk_text in enumerate(text_chunks):
            # Each chunk inherits ALL parent metadata + gets chunk_index
            chunk_metadata = {
                **metadata,            # spread parent metadata (source, title, url, etc.)
                "chunk_index": i,      # position within the original document
                "total_chunks": len(text_chunks),  # how many chunks this doc produced
            }

            all_chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata,
            })

    return all_chunks


def print_chunk_stats(chunks: list[dict]) -> None:
    """
    Print statistics about the chunking results.

    Useful for sanity-checking:
    - Are chunks the right size? (not too big, not too small)
    - How many chunks per document? (too many = chunks are too small)
    - Any outliers? (one doc producing 50 chunks = something might be wrong)
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    sizes = [len(c["content"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes) if sizes else 0

    # Count chunks per source document
    docs_count: dict[str, int] = {}
    for chunk in chunks:
        title = chunk["metadata"]["title"]
        docs_count[title] = docs_count.get(title, 0) + 1

    table = Table(title="📊 Chunking Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Total chunks", str(len(chunks)))
    table.add_row("Avg chunk size", f"{avg_size:.0f} chars")
    table.add_row("Min chunk size", f"{min(sizes)} chars")
    table.add_row("Max chunk size", f"{max(sizes)} chars")
    table.add_row("Source documents", str(len(docs_count)))
    table.add_row(
        "Biggest doc (by chunks)",
        f"{max(docs_count, key=docs_count.get)} ({max(docs_count.values())} chunks)",
    )

    console.print(table)


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: uv run python -m src.chunking.markdown_chunker

    Uses a sample markdown document (no GitHub API call needed).
    Shows how the splitter breaks it into chunks.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    # Sample markdown document (simulates a FastAPI doc page)
    sample_doc = {
        "content": """## CORS (Cross-Origin Resource Sharing)

You can configure CORS in your FastAPI application using the `CORSMiddleware`.

### What is CORS?

CORS is a mechanism that allows a web page to make requests to a different domain
than the one that served the page. This is important when your frontend (React, Vue)
is served from a different origin than your API.

For example, if your frontend is at `http://localhost:3000` and your API is at
`http://localhost:8000`, the browser will block requests unless CORS is configured.

### How to use it

First, import `CORSMiddleware`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://myapp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Configuration options

The `CORSMiddleware` accepts several arguments:

- `allow_origins`: List of origins that are allowed to make requests.
- `allow_methods`: HTTP methods allowed for cross-origin requests.
- `allow_headers`: HTTP headers allowed in cross-origin requests.
- `allow_credentials`: Whether cookies should be supported.

### More information

You can read more about CORS in the MDN Web Docs:
https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS""",
        "metadata": {
            "source": "docs",
            "title": "cors",
            "url": "https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/cors.md",
            "file_path": "docs/en/docs/tutorial/cors.md",
        },
    }

    console.print("\n[bold]✂️  Testing Markdown Chunker[/bold]\n")
    console.print(f"Original document: {len(sample_doc['content'])} chars\n")

    chunks = chunk_documents([sample_doc])

    for i, chunk in enumerate(chunks):
        console.print(Panel(
            chunk["content"][:300] + ("..." if len(chunk["content"]) > 300 else ""),
            title=f"Chunk {i} ({len(chunk['content'])} chars)",
            subtitle=f"metadata: source={chunk['metadata']['source']}, "
                     f"title={chunk['metadata']['title']}, "
                     f"chunk_index={chunk['metadata']['chunk_index']}",
        ))

    print_chunk_stats(chunks)