"""
Hybrid Retriever — combines Dense + Sparse + Reranking.

This is the full advanced retrieval pipeline:

  ┌──────────────┐     ┌──────────────┐
  │   Pinecone   │     │    BM25      │
  │   (Dense)    │     │   (Sparse)   │
  │              │     │              │
  │ Understands  │     │ Exact keyword│
  │ meaning      │     │ matching     │
  └──────┬───────┘     └──────┬───────┘
         │                    │
         ▼                    ▼
  ┌──────────────────────────────────┐
  │   Reciprocal Rank Fusion (RRF)  │
  │                                  │
  │   Merges both ranked lists.     │
  │   Docs appearing in BOTH lists  │
  │   get boosted.                  │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │   Cohere Reranker               │
  │                                  │
  │   Cross-encoder reads each      │
  │   (query, candidate) pair.      │
  │   Produces final ranking.       │
  └──────────────┬───────────────────┘
                 │
                 ▼
           Top K results

Why this order?
  1. Dense + Sparse cast a wide net (high recall)
  2. RRF merges and deduplicates
  3. Reranker sorts by true relevance (high precision)
"""

from rich.console import Console

from src.retrieval.pinecone_retriever import PineconeRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CohereReranker

console = Console()


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF formula:
      RRF_score(doc) = Σ  1 / (k + rank_in_list_i)

    For each document, sum its RRF contribution from every list
    it appears in. Documents appearing in multiple lists get
    higher combined scores.

    Parameters:
    - ranked_lists: list of ranked result lists
      e.g., [dense_results, sparse_results]
    - k: smoothing constant (default 60, from the original RRF paper)
      Higher k = less weight to top-ranked items
      60 is the standard value used in the original research paper.

    Returns: merged and re-scored list, sorted by RRF score.

    Example:
      Dense results:  [A(rank 1), B(rank 2), C(rank 3)]
      Sparse results: [C(rank 1), D(rank 2), A(rank 3)]

      RRF scores:
        A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
        C: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
        B: 1/(60+2)            = 0.0161
        D: 1/(60+2)            = 0.0161

      Result: [A, C, B, D]  (A and C tied — both in both lists)
    """
    # Track RRF scores by document ID (we use text content as key
    # since not all results have the same ID format)
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            # Use text content as unique key (first 200 chars for efficiency)
            doc_key = result["text"][:200]

            # Add RRF contribution
            rrf_score = 1.0 / (k + rank + 1)  # rank+1 because rank is 0-indexed
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + rrf_score

            # Keep the result dict (use the one with better original score)
            if doc_key not in doc_map or result["score"] > doc_map[doc_key]["score"]:
                doc_map[doc_key] = result

    # Sort by RRF score
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Build final list with RRF scores
    merged = []
    for key in sorted_keys:
        result = doc_map[key].copy()
        result["rrf_score"] = rrf_scores[key]
        merged.append(result)

    return merged


class HybridRetriever:
    """
    Full advanced retrieval pipeline: Dense + Sparse + RRF + Reranking.

    Usage:
        retriever = HybridRetriever()
        results = retriever.retrieve("How do I add CORS?")
    """

    def __init__(
        self,
        top_k: int = 5,
        dense_top_k: int = 15,
        sparse_top_k: int = 15,
        use_reranker: bool = True,
    ):
        """
        Initialize the hybrid retriever.

        Parameters:
        - top_k: final number of results to return
        - dense_top_k: how many candidates to fetch from Pinecone
        - sparse_top_k: how many candidates to fetch from BM25
        - use_reranker: whether to apply Cohere reranking

        Why fetch 15 candidates from each retriever?
        We want a large candidate pool for RRF and reranking to
        work with. After merging and deduplication, we'll have
        ~20-25 unique candidates. The reranker picks the best 5.
        """
        self.top_k = top_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k

        console.print("\n[bold]Initializing Hybrid Retriever...[/bold]")

        # Initialize components
        self.dense_retriever = PineconeRetriever(
            top_k=dense_top_k,
            use_mmr=False,  # skip MMR — reranker handles diversity better
        )
        self.sparse_retriever = BM25Retriever()
        self.reranker = CohereReranker() if use_reranker else None

        console.print("[green]✅ Hybrid retriever ready[/green]\n")

    def retrieve(self, query: str) -> list[dict]:
        """
        Full retrieval pipeline.

        Steps:
        1. Dense search (Pinecone) → 15 candidates
        2. Sparse search (BM25) → 15 candidates
        3. RRF merge → ~20-25 unique candidates
        4. Cohere rerank → top 5 (final answer)

        Returns list of result dicts, same shape as PineconeRetriever.
        """
        # Step 1: Dense retrieval (semantic search)
        dense_results = self.dense_retriever.retrieve(query)

        # Step 2: Sparse retrieval (keyword search)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=self.sparse_top_k)

        # Step 3: Merge with RRF
        merged = reciprocal_rank_fusion([dense_results, sparse_results])

        # Step 4: Rerank (if enabled)
        if self.reranker and self.reranker.client:
            # Send more candidates to reranker than we need (let it pick the best)
            candidates_for_reranking = merged[:20]
            final_results = self.reranker.rerank(
                query=query,
                results=candidates_for_reranking,
                top_k=self.top_k,
            )
        else:
            final_results = merged[:self.top_k]

        return final_results

    def retrieve_dense_only(self, query: str) -> list[dict]:
        """Retrieve using only Pinecone (for comparison)."""
        return self.dense_retriever.retrieve(query)[:self.top_k]

    def retrieve_sparse_only(self, query: str) -> list[dict]:
        """Retrieve using only BM25 (for comparison)."""
        return self.sparse_retriever.retrieve(query, top_k=self.top_k)

    def retrieve_without_reranking(self, query: str) -> list[dict]:
        """Retrieve with hybrid but without reranking (for comparison)."""
        dense_results = self.dense_retriever.retrieve(query)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=self.sparse_top_k)
        merged = reciprocal_rank_fusion([dense_results, sparse_results])
        return merged[:self.top_k]


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.panel import Panel

    console.print("\n[bold]🔀 Testing Hybrid Retriever[/bold]\n")

    retriever = HybridRetriever(top_k=3)

    query = "422 error with Pydantic model validation"
    console.print(f"[bold]Query:[/bold] {query}\n")

    results = retriever.retrieve(query)

    for i, r in enumerate(results, 1):
        preview = r["text"][:200].replace("\n", " ")
        rerank = f", rerank={r['rerank_score']:.4f}" if "rerank_score" in r else ""
        console.print(Panel(
            f"[bold]Scores:[/bold] rrf={r.get('rrf_score', 0):.4f}{rerank}\n"
            f"[bold]Source:[/bold] {r.get('namespace', 'unknown')}\n"
            f"[bold]Title:[/bold] {r['metadata'].get('title', 'unknown')}\n"
            f"[bold]Preview:[/bold] {preview}...",
            title=f"Hybrid Result #{i}",
        ))