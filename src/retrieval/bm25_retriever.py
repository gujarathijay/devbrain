"""
BM25 Sparse Retriever — keyword-based search.

BM25 (Best Matching 25) is a ranking function based on term frequency.
Unlike semantic search (Pinecone), BM25 finds documents containing
the exact words from your query.

When BM25 beats semantic search:
  - Error codes: "422", "CORS", "EBITDA"
  - Function names: "Depends()", "BackgroundTasks"
  - Specific terms: "OAuth2PasswordBearer"

When semantic search beats BM25:
  - Meaning-based queries: "how to validate request data"
    (BM25 won't find "Pydantic" unless the word appears in the query)
  - Paraphrased concepts: "making API calls faster"
    (BM25 won't find docs about "caching" or "async")

That's why we use BOTH — they complement each other.

How BM25 scoring works:
  For each term in the query:
    score += IDF(term) × (TF(term, doc) × (k1 + 1)) / (TF(term, doc) + k1 × ...)

  Where:
    TF  = how often the term appears in this document (more = better)
    IDF = how rare the term is across ALL documents (rarer = more valuable)
    k1  = controls how much term frequency matters (default 1.5)

  Example:
    Query: "422 error Pydantic"
    Doc A: contains "422" 3 times, "error" 5 times, "Pydantic" 2 times → HIGH score
    Doc B: contains "error" 10 times, no "422", no "Pydantic" → LOW score
    ("error" alone isn't enough — the rare terms "422" and "Pydantic" matter more)
"""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from rich.console import Console

from src.config import RAW_DOCS_DIR, RAW_ISSUES_DIR
from src.chunking.markdown_chunker import chunk_documents
from src.chunking.issue_chunker import chunk_issues

console = Console()


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.

    BM25 works on individual words (tokens). We:
    1. Lowercase everything (so "CORS" matches "cors")
    2. Split on whitespace and punctuation
    3. Remove very short tokens (single chars)

    This is intentionally simple. Production BM25 systems use
    more sophisticated tokenizers with stemming (reducing "running"
    to "run") and stop word removal (removing "the", "is", "a").
    For our use case, simple splitting works well enough.
    """
    # Replace common punctuation with spaces, then split
    for char in ".,;:!?()[]{}\"'`/\\|<>=+*&^%$#@~":
        text = text.replace(char, " ")
    tokens = text.lower().split()
    # Keep tokens with at least 2 characters
    return [t for t in tokens if len(t) >= 2]


class BM25Retriever:
    """
    BM25-based sparse retriever.

    This builds an in-memory BM25 index from the raw data.
    It re-chunks the data on initialization (takes a few seconds)
    to ensure the chunks match what's in Pinecone.

    Usage:
        retriever = BM25Retriever()
        results = retriever.retrieve("422 error with Pydantic", top_k=10)
    """

    def __init__(self):
        """
        Initialize by loading raw data, chunking it, and building the BM25 index.

        Why re-chunk instead of loading from Pinecone?
        - BM25 needs the raw text + metadata locally
        - Re-chunking is fast (~2 seconds, no API calls)
        - Ensures BM25 chunks match Pinecone chunks exactly
        - Pinecone's free tier doesn't support fetching all vectors
        """
        console.print("[dim]Building BM25 index...[/dim]")

        # Load and chunk docs
        self.chunks: list[dict] = []
        doc_documents = self._load_raw_docs()
        if doc_documents:
            self.chunks.extend(chunk_documents(doc_documents))

        # Load and chunk issues
        issue_documents = self._load_raw_issues()
        if issue_documents:
            self.chunks.extend(chunk_issues(issue_documents))

        if not self.chunks:
            console.print("[red]⚠️  No chunks to index. Run ingestion first.[/red]")
            self._index = None
            return

        # Tokenize all chunks for BM25
        self._tokenized_corpus = [_tokenize(c["content"]) for c in self.chunks]

        # Build the BM25 index
        # BM25Okapi is the standard BM25 variant (Okapi BM25)
        # It takes a list of tokenized documents
        self._index = BM25Okapi(self._tokenized_corpus)

        console.print(
            f"[green]✅ BM25 index built: {len(self.chunks)} chunks[/green]"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        namespace: str | None = None,
    ) -> list[dict]:
        """
        Search using BM25.

        Parameters:
        - query: the user's question (plain text)
        - top_k: how many results to return
        - namespace: optional filter ("docs" or "issues")

        Returns list of dicts matching the same shape as PineconeRetriever:
          [
            {
              "text": "chunk content...",
              "score": 8.54,            ← BM25 score (not 0-1, unlike cosine)
              "metadata": {...},
              "namespace": "docs"
            },
            ...
          ]

        Note: BM25 scores are NOT between 0 and 1. They can be any
        positive number. A score of 15 is better than 5, but the
        absolute values don't mean the same as cosine similarity.
        RRF handles this by using RANK position, not raw scores.
        """
        if self._index is None:
            return []

        # Tokenize the query the same way we tokenized documents
        query_tokens = _tokenize(query)

        # Get BM25 scores for all documents
        scores = self._index.get_scores(query_tokens)

        # Build result list with scores
        results = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue  # skip documents with zero relevance

            chunk = self.chunks[i]
            source = chunk["metadata"].get("source", "")

            # Apply namespace filter if specified
            if namespace and source != namespace:
                continue

            results.append({
                "text": chunk["content"],
                "score": float(score),
                "metadata": chunk["metadata"],
                "namespace": source,
            })

        # Sort by score (highest first) and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _load_raw_docs(self) -> list[dict]:
        """Load raw markdown docs from disk (same logic as run_ingestion.py)."""
        documents = []
        if not RAW_DOCS_DIR.exists():
            return []

        for md_file in sorted(RAW_DOCS_DIR.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            if len(content.strip()) < 50:
                continue
            relative_path = md_file.relative_to(RAW_DOCS_DIR)
            documents.append({
                "content": content,
                "metadata": {
                    "source": "docs",
                    "file_path": str(relative_path),
                    "title": md_file.stem,
                    "url": f"https://github.com/fastapi/fastapi/blob/master/docs/en/docs/{relative_path}",
                },
            })
        return documents

    def _load_raw_issues(self) -> list[dict]:
        """Load raw issues from disk."""
        documents = []
        if not RAW_ISSUES_DIR.exists():
            return []

        for json_file in sorted(RAW_ISSUES_DIR.glob("issue_*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if len(data.get("content", "").strip()) < 50:
                continue
            documents.append(data)
        return documents


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.panel import Panel

    console.print("\n[bold]🔍 Testing BM25 Retriever[/bold]\n")

    retriever = BM25Retriever()

    # Test with a keyword-heavy query (BM25's strength)
    query = "422 Unprocessable Entity error"
    console.print(f"[bold]Query:[/bold] {query}\n")

    results = retriever.retrieve(query, top_k=3)

    for i, r in enumerate(results, 1):
        preview = r["text"][:200].replace("\n", " ")
        console.print(Panel(
            f"[bold]Score:[/bold] {r['score']:.2f}\n"
            f"[bold]Source:[/bold] {r['namespace']}\n"
            f"[bold]Title:[/bold] {r['metadata'].get('title', 'unknown')}\n"
            f"[bold]Preview:[/bold] {preview}...",
            title=f"BM25 Result #{i}",
        ))