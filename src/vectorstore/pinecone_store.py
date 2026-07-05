"""
Pinecone vector store — index creation, upserting, and querying.

This module handles all interaction with Pinecone:
  1. Create the index (if it doesn't exist)
  2. Upsert vectors (store embedded chunks with metadata)
  3. Query vectors (find similar chunks for a given question)
  4. Get stats (how many vectors per namespace)

Pinecone vocabulary:
  - Index:     like a database table. We have one: "devbrain"
  - Namespace: like folders inside the index. We have "docs" and "issues"
  - Vector:    one embedded chunk (1536 numbers + metadata)
  - Upsert:    insert-or-update. If the ID already exists, it overwrites.
  - Query:     send a vector, get back the top-K most similar vectors

Why Pinecone (vs storing vectors in a local file)?
  - Optimized for similarity search at scale (millions of vectors)
  - Metadata filtering (search only "docs" from "2024")
  - Managed infrastructure (no server to maintain)
  - Free tier is enough for our project (2GB, ~100K vectors)
"""

import time

from pinecone import Pinecone, ServerlessSpec
from rich.console import Console

from src.config import (
    EMBEDDING_DIMENSION,
    NAMESPACE_DOCS,
    NAMESPACE_ISSUES,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
)

console = Console()

# ── Initialize Pinecone client ────────────────────────────────────
# Like the OpenAI client, we create one instance and reuse it.
pc = Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index() -> object:
    """
    Get the Pinecone index, creating it if it doesn't exist.

    ServerlessSpec explained:
      - cloud="aws": Pinecone runs on AWS infrastructure
      - region="us-east-1": the AWS region (free tier uses us-east-1)

    The index is created with:
      - dimension=1536: must match our embedding model's output dimension
        (text-embedding-3-small = 1536). If these don't match, upserts will fail.
      - metric="cosine": how similarity is measured between vectors.
        Cosine similarity is standard for text embeddings because it
        measures the ANGLE between vectors, not their magnitude.
        This means a short text and a long text about the same topic
        will still score as similar.

    Why we wait after creation:
      Pinecone creates indexes asynchronously. The API call returns
      immediately, but the index isn't ready yet. We poll until
      it's ready (usually 30-60 seconds for serverless).
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME in existing_indexes:
        console.print(f"[green]✅ Index '{PINECONE_INDEX_NAME}' already exists[/green]")
    else:
        console.print(f"[yellow]Creating index '{PINECONE_INDEX_NAME}'...[/yellow]")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait for the index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            console.print("  Waiting for index to be ready...")
            time.sleep(5)

        console.print(f"[green]✅ Index '{PINECONE_INDEX_NAME}' created![/green]")

    return pc.Index(PINECONE_INDEX_NAME)


def upsert_chunks(
    index: object,
    chunks: list[dict],
    namespace: str,
    batch_size: int = 100,
) -> int:
    """
    Upsert embedded chunks into a Pinecone namespace.

    Each chunk dict must have:
      - "content":   the raw text
      - "metadata":  dict with source, title, url, etc.
      - "embedding": list of 1536 floats

    We convert each chunk into a Pinecone vector:
      {
        "id": "docs_cors_0",           ← unique ID
        "values": [0.023, -0.041, ...], ← the embedding
        "metadata": {
          "text": "## CORS...",         ← raw text for retrieval
          "source": "docs",            ← from the chunk metadata
          "title": "cors",
          ...
        }
      }

    Why "text" in metadata?
      Pinecone stores vectors but doesn't inherently store the original
      text. When we query and get back a match, we need the actual text
      to show the user. So we store it in metadata under the key "text".

    Why batch upserts?
      Pinecone's upsert API accepts up to ~100 vectors per call.
      Sending them one-by-one would be 3000 API calls vs 30.

    Parameters:
    - index: the Pinecone index object
    - chunks: list of chunk dicts with embeddings
    - namespace: "docs" or "issues"
    - batch_size: vectors per upsert call

    Returns: number of vectors upserted
    """
    vectors = []
    for i, chunk in enumerate(chunks):
        # Build a unique ID: namespace_title_chunkindex
        # e.g., "docs_cors_0", "docs_cors_1", "issues_9876_0"
        title_slug = chunk["metadata"].get("title", "unknown")
        # Clean the title for use as an ID (remove special chars)
        title_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(title_slug))
        vector_id = f"{namespace}_{title_slug}_{chunk['metadata']['chunk_index']}"

        # Build metadata for Pinecone
        # We store the raw text so we can retrieve it later
        pinecone_metadata = {
            "text": chunk["content"][:8000],  # Pinecone metadata limit: ~40KB
            "source": chunk["metadata"].get("source", ""),
            "title": str(chunk["metadata"].get("title", "")),
            "url": chunk["metadata"].get("url", ""),
            "chunk_index": chunk["metadata"].get("chunk_index", 0),
        }

        # Add issue-specific metadata if present
        if "issue_number" in chunk["metadata"]:
            pinecone_metadata["issue_number"] = chunk["metadata"]["issue_number"]
            pinecone_metadata["state"] = chunk["metadata"].get("state", "")
            # Pinecone metadata values must be strings, numbers, booleans, or lists of strings
            pinecone_metadata["labels"] = chunk["metadata"].get("labels", [])

        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": pinecone_metadata,
        })

    # Upsert in batches
    upserted = 0
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        total_batches = (len(vectors) + batch_size - 1) // batch_size
        task = progress.add_task(
            f"Upserting to '{namespace}'...", total=total_batches
        )

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            upserted += len(batch)
            progress.advance(task)

            # Small delay to avoid rate limits
            time.sleep(0.1)

    console.print(
        f"[green]✅ Upserted {upserted} vectors to "
        f"namespace '{namespace}'[/green]"
    )
    return upserted


def query_index(
    index: object,
    query_embedding: list[float],
    namespace: str,
    top_k: int = 5,
    filter_dict: dict | None = None,
) -> list[dict]:
    """
    Query Pinecone for the most similar vectors.

    Parameters:
    - query_embedding: the embedded question (1536 floats)
    - namespace: which namespace to search ("docs", "issues", or "")
    - top_k: how many results to return
    - filter_dict: optional metadata filter
      Example: {"state": "closed"} → only return closed issues
      Example: {"source": "docs"} → only return docs (redundant with namespace)

    Returns a list of result dicts:
      [
        {
          "id": "docs_cors_0",
          "score": 0.92,           ← cosine similarity (0 to 1)
          "text": "## CORS...",     ← the actual chunk text
          "metadata": {...}         ← all stored metadata
        },
        ...
      ]

    The results are sorted by score (highest = most similar = first).
    """
    query_params = {
        "vector": query_embedding,
        "top_k": top_k,
        "namespace": namespace,
        "include_metadata": True,
    }

    if filter_dict:
        query_params["filter"] = filter_dict

    results = index.query(**query_params)

    # Convert Pinecone response to clean dicts
    formatted = []
    for match in results.matches:
        formatted.append({
            "id": match.id,
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "metadata": dict(match.metadata),
        })

    return formatted


def get_index_stats(index: object) -> dict:
    """
    Get statistics about the Pinecone index.

    Returns info like:
      - Total vector count
      - Vectors per namespace
      - Index fullness (how much of the free tier we're using)

    Useful for verifying that upserts worked correctly.
    """
    stats = index.describe_index_stats()

    return {
        "total_vectors": stats.total_vector_count,
        "namespaces": {
            ns: data.vector_count
            for ns, data in stats.namespaces.items()
        },
        "dimension": stats.dimension,
    }


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: uv run python -m src.vectorstore.pinecone_store

    Creates the index (if needed) and prints its stats.
    Does NOT upsert any data — that's done by the pipeline script.
    """
    from rich.table import Table

    console.print("\n[bold]🌲 Testing Pinecone Connection[/bold]\n")

    # Create or get index
    index = get_or_create_index()

    # Get stats
    stats = get_index_stats(index)

    table = Table(title="Pinecone Index Stats")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Index name", PINECONE_INDEX_NAME)
    table.add_row("Dimension", str(stats["dimension"]))
    table.add_row("Total vectors", str(stats["total_vectors"]))

    for ns, count in stats["namespaces"].items():
        table.add_row(f"Namespace: {ns}", f"{count} vectors")

    if not stats["namespaces"]:
        table.add_row("Namespaces", "(empty — no data yet)")

    console.print(table)
    console.print("\n[green]✅ Pinecone connection working![/green]")