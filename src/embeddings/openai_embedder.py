"""
Batch embedding using the OpenAI Embeddings API.

This module takes a list of text chunks and returns their vector embeddings.
It handles:
  - Batching (sending 100 texts per API call instead of one-by-one)
  - Rate limiting (waiting + retrying on 429 errors)
  - Error handling (clear messages when things go wrong)
  - Cost estimation (so you know what you're spending)

How OpenAI embedding pricing works (text-embedding-3-small):
  $0.02 per 1 million tokens
  Average chunk ≈ 250 tokens
  1,000 chunks ≈ 250,000 tokens ≈ $0.005
  5,000 chunks ≈ 1,250,000 tokens ≈ $0.025
  Extremely cheap — this whole pipeline costs less than a coffee.
"""

import time

import openai
import tiktoken
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, OPENAI_API_KEY

console = Console()

# ── Initialize OpenAI client ─────────────────────────────────────
# We create ONE client instance and reuse it for all calls.
# This is more efficient than creating a new client per request
# because the client manages connection pooling internally.
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ── Token counter ─────────────────────────────────────────────────
# tiktoken is OpenAI's tokenizer — it counts tokens the same way
# the API does, so our cost estimates are accurate.
_encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    """Count how many tokens a text will use."""
    return len(_encoding.encode(text))


def estimate_cost(texts: list[str]) -> dict:
    """
    Estimate the API cost for embedding a list of texts.

    Returns a dict with token count and estimated cost in USD.
    Useful to run BEFORE the actual embedding call so you know
    what you're about to spend.
    """
    total_tokens = sum(count_tokens(t) for t in texts)
    # text-embedding-3-small: $0.02 per 1M tokens
    cost_usd = (total_tokens / 1_000_000) * 0.02

    return {
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
        "num_texts": len(texts),
        "avg_tokens_per_text": total_tokens // len(texts) if texts else 0,
    }


def embed_texts(
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_retries: int = 5,
) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI's embedding API.

    Parameters:
    - texts: list of strings to embed
    - batch_size: how many texts to send per API call (default 100)
    - max_retries: how many times to retry on rate limit errors

    Returns:
    - list of embeddings, each a list of 1536 floats
    - Order is preserved: embeddings[i] corresponds to texts[i]

    How batching works:
      texts = ["text1", "text2", ..., "text250"]

      Batch 1: embed texts[0:100]   → 100 embeddings
      Batch 2: embed texts[100:200] → 100 embeddings
      Batch 3: embed texts[200:250] → 50 embeddings

      Total: 3 API calls instead of 250
    """
    all_embeddings: list[list[float]] = []

    # Calculate total batches for progress bar
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding...", total=total_batches)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = _embed_batch_with_retry(batch, max_retries)
            all_embeddings.extend(embeddings)
            progress.advance(task)

    return all_embeddings


def _embed_batch_with_retry(
    texts: list[str],
    max_retries: int = 5,
) -> list[list[float]]:
    """
    Embed a single batch with exponential backoff retry.

    Why retry with backoff?
    The OpenAI API has rate limits. If we hit them:
      - First retry: wait 1 second
      - Second retry: wait 2 seconds
      - Third retry: wait 4 seconds
      - Fourth retry: wait 8 seconds
      - Fifth retry: wait 16 seconds
      - After that: give up and raise the error

    The wait time doubles each attempt (exponential backoff).
    This is the standard pattern for rate-limited APIs because:
    1. Short waits handle brief rate limit windows
    2. Longer waits handle sustained overload
    3. We don't hammer the server with rapid retries
    """
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )

            # The API returns embeddings in the same order as input.
            # response.data is a list of Embedding objects, each with
            # an .embedding attribute (the vector) and an .index attribute.
            #
            # We sort by index to be safe (the API guarantees order,
            # but being defensive costs nothing).
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except openai.RateLimitError:
            # 429 Too Many Requests — we're sending too fast
            wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
            console.print(
                f"[yellow]⚠️  Rate limited. Waiting {wait_time}s "
                f"(attempt {attempt + 1}/{max_retries})...[/yellow]"
            )
            time.sleep(wait_time)

        except openai.APIError as e:
            # Other API errors (500 server error, etc.)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                console.print(
                    f"[yellow]⚠️  API error: {e}. Retrying in {wait_time}s...[/yellow]"
                )
                time.sleep(wait_time)
            else:
                raise  # give up after max retries

    # If we exhausted all retries (only reached via RateLimitError path)
    raise RuntimeError(
        f"Failed to embed batch after {max_retries} retries. "
        f"Check your OpenAI API key and rate limits."
    )


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    High-level function: takes chunk dicts, returns chunk dicts with embeddings added.

    This is the main entry point for the pipeline. It:
    1. Extracts text from chunk dicts
    2. Estimates cost and prints it
    3. Embeds all texts in batches
    4. Attaches the embedding vector to each chunk dict

    Input:
      [{"content": "## CORS...", "metadata": {...}}, ...]

    Output:
      [{"content": "## CORS...", "metadata": {...}, "embedding": [0.023, ...]}, ...]
    """
    texts = [chunk["content"] for chunk in chunks]

    # Show cost estimate before embedding
    cost = estimate_cost(texts)
    console.print(
        f"  Embedding {cost['num_texts']} chunks "
        f"({cost['total_tokens']:,} tokens, "
        f"~${cost['estimated_cost_usd']:.4f})"
    )

    # Embed all texts
    embeddings = embed_texts(texts)

    # Attach embeddings to chunk dicts
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    console.print(f"[green]✅ Embedded {len(chunks)} chunks[/green]")
    return chunks


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: uv run python -m src.embeddings.openai_embedder

    Embeds 3 sample texts and shows their similarity scores.
    This verifies your OpenAI API key works and shows how
    cosine similarity captures meaning.
    """
    import numpy as np

    console.print("\n[bold]🔢 Testing OpenAI Embedder[/bold]\n")

    # Three test texts — two are similar, one is different
    test_texts = [
        "How do I add CORS middleware to FastAPI?",
        "Adding CORS headers in a FastAPI application",
        "What is the capital of France?",
    ]

    # Estimate cost first
    cost = estimate_cost(test_texts)
    console.print(f"  Estimated cost: ${cost['estimated_cost_usd']:.6f}")
    console.print(f"  Total tokens: {cost['total_tokens']}\n")

    # Embed
    embeddings = embed_texts(test_texts)

    console.print(f"  Got {len(embeddings)} embeddings")
    console.print(f"  Each embedding has {len(embeddings[0])} dimensions\n")

    # Calculate cosine similarity between pairs
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """
        Cosine similarity = dot(a, b) / (||a|| * ||b||)

        Returns a score between -1 and 1:
          1.0 = identical meaning
          0.0 = completely unrelated
         -1.0 = opposite meaning (rare in practice)
        """
        a_arr, b_arr = np.array(a), np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    console.print("[bold]Cosine Similarity Scores:[/bold]")
    console.print(f'  "{test_texts[0]}"')
    console.print(f'  vs "{test_texts[1]}"')
    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    console.print(f"  → [green]{sim_01:.4f}[/green] (should be HIGH — same topic)\n")

    console.print(f'  "{test_texts[0]}"')
    console.print(f'  vs "{test_texts[2]}"')
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])
    console.print(f"  → [red]{sim_02:.4f}[/red] (should be LOW — different topics)\n")