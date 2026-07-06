"""
Pinecone Retriever — wraps Pinecone queries into a reusable component.

This module provides a clean interface for retrieving relevant chunks
from Pinecone. It handles:
  - Embedding the user's question
  - Querying one or multiple namespaces
  - Returning results in a consistent format
  - MMR (Max Marginal Relevance) for diverse results

Why wrap Pinecone in a retriever class?
  - The RAG chain doesn't need to know about Pinecone internals
  - Easy to swap in a different vector store later (Chroma, Weaviate, etc.)
  - Encapsulates the embed → query → format pipeline
  - In Phase 3, we'll add hybrid search and reranking here
"""

import numpy as np

from src.config import NAMESPACE_DOCS, NAMESPACE_ISSUES
from src.embeddings.openai_embedder import embed_texts
from src.vectorstore.pinecone_store import get_or_create_index, query_index


class PineconeRetriever:
    """
    Retrieves relevant chunks from Pinecone for a given question.

    Usage:
        retriever = PineconeRetriever()
        results = retriever.retrieve("How do I add CORS?")
        # returns list of {"text": "...", "score": 0.87, "metadata": {...}}
    """

    def __init__(
        self,
        top_k: int = 5,
        namespaces: list[str] | None = None,
        use_mmr: bool = True,
        mmr_diversity: float = 0.3,
    ):
        """
        Initialize the retriever.

        Parameters:
        - top_k: how many results to return (default 5)
        - namespaces: which namespaces to search (default: both docs and issues)
        - use_mmr: whether to apply MMR for diversity (default True)
        - mmr_diversity: how much to prioritize diversity vs relevance (0-1)
          0.0 = pure relevance (same as no MMR)
          1.0 = pure diversity (ignores relevance)
          0.3 = good balance (our default)
        """
        self.top_k = top_k
        self.namespaces = namespaces or [NAMESPACE_DOCS, NAMESPACE_ISSUES]
        self.use_mmr = use_mmr
        self.mmr_diversity = mmr_diversity
        self.index = get_or_create_index()

    def retrieve(self, question: str) -> list[dict]:
        """
        Retrieve relevant chunks for a question.

        Steps:
        1. Embed the question using the same model used for chunks
           (this is critical — question and chunks must use the same
           embedding model, otherwise similarity scores are meaningless)
        2. Query each namespace
        3. Merge results from all namespaces
        4. Apply MMR if enabled (for diversity)
        5. Return top_k results sorted by score

        Returns list of dicts:
          [
            {
              "text": "## CORS...",
              "score": 0.87,
              "metadata": {"source": "docs", "title": "cors", ...},
              "namespace": "docs"
            },
            ...
          ]
        """
        # Step 1: Embed the question
        question_embedding = embed_texts([question])[0]

        # Step 2 & 3: Query each namespace and merge
        all_results = []

        for namespace in self.namespaces:
            # Fetch more than top_k per namespace so MMR has candidates to choose from
            fetch_k = self.top_k * 3 if self.use_mmr else self.top_k

            results = query_index(
                index=self.index,
                query_embedding=question_embedding,
                namespace=namespace,
                top_k=fetch_k,
            )

            # Add namespace info to each result
            for r in results:
                r["namespace"] = namespace

            all_results.extend(results)

        if not all_results:
            return []

        # Step 4: Apply MMR or simple sorting
        if self.use_mmr and len(all_results) > self.top_k:
            selected = self._apply_mmr(
                query_embedding=question_embedding,
                results=all_results,
                k=self.top_k,
                lambda_param=1 - self.mmr_diversity,
            )
        else:
            # Simple: sort by score, take top_k
            all_results.sort(key=lambda x: x["score"], reverse=True)
            selected = all_results[: self.top_k]

        return selected

    def retrieve_from_namespace(
        self, question: str, namespace: str, top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve from a specific namespace only.

        Useful when you know the question type:
        - How-to question → search docs only
        - Error/bug question → search issues only

        In Phase 4, the LangGraph agent will use this to route
        queries to the right namespace.
        """
        k = top_k or self.top_k
        question_embedding = embed_texts([question])[0]

        results = query_index(
            index=self.index,
            query_embedding=question_embedding,
            namespace=namespace,
            top_k=k,
        )

        for r in results:
            r["namespace"] = namespace

        return results

    def _apply_mmr(
        self,
        query_embedding: list[float],
        results: list[dict],
        k: int,
        lambda_param: float = 0.7,
    ) -> list[dict]:
        """
        Apply Max Marginal Relevance to select diverse results.

        MMR formula for each candidate:
          MMR_score = lambda * similarity(query, candidate)
                    - (1 - lambda) * max(similarity(candidate, already_selected))

        In plain English:
          "Pick candidates that are similar to the question (first term)
           but different from what we already picked (second term)"

        lambda_param controls the balance:
          - lambda=1.0: pure relevance (ignores diversity)
          - lambda=0.5: equal balance
          - lambda=0.7: favor relevance, but still encourage diversity (our default)

        Algorithm:
          1. Start with the highest-scoring result
          2. For each remaining slot:
             a. For each candidate, calculate MMR score
             b. Pick the candidate with highest MMR score
             c. Move it from candidates to selected
          3. Return selected list
        """
        if not results:
            return []

        query_vec = np.array(query_embedding)

        # We need embeddings for MMR calculation, but Pinecone results
        # don't include the vector values. We'll re-embed the result texts.
        # This costs a tiny amount (~$0.00001) but gives us accurate MMR.
        result_texts = [r["text"] for r in results]
        result_embeddings = embed_texts(result_texts)
        result_vecs = np.array(result_embeddings)

        # Calculate query-to-result similarities
        query_sims = np.dot(result_vecs, query_vec) / (
            np.linalg.norm(result_vecs, axis=1) * np.linalg.norm(query_vec)
        )

        # Greedy MMR selection
        selected_indices: list[int] = []
        candidate_indices = list(range(len(results)))

        # First pick: highest similarity to query
        best_idx = int(np.argmax(query_sims))
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

        # Remaining picks: balance relevance vs diversity
        while len(selected_indices) < k and candidate_indices:
            best_score = -float("inf")
            best_candidate = -1

            for candidate_idx in candidate_indices:
                # Relevance: similarity to the query
                relevance = query_sims[candidate_idx]

                # Redundancy: max similarity to any already-selected result
                if selected_indices:
                    selected_vecs = result_vecs[selected_indices]
                    candidate_vec = result_vecs[candidate_idx]
                    similarities = np.dot(selected_vecs, candidate_vec) / (
                        np.linalg.norm(selected_vecs, axis=1)
                        * np.linalg.norm(candidate_vec)
                    )
                    redundancy = float(np.max(similarities))
                else:
                    redundancy = 0.0

                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate_idx

            if best_candidate >= 0:
                selected_indices.append(best_candidate)
                candidate_indices.remove(best_candidate)
            else:
                break

        return [results[i] for i in selected_indices]


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    console.print("\n[bold]🔍 Testing Pinecone Retriever[/bold]\n")

    retriever = PineconeRetriever(top_k=3, use_mmr=True)

    question = "How do I add authentication to FastAPI?"
    console.print(f"[bold]Question:[/bold] {question}\n")

    results = retriever.retrieve(question)

    for i, r in enumerate(results, 1):
        preview = r["text"][:200].replace("\n", " ")
        console.print(Panel(
            f"[bold]Score:[/bold] {r['score']:.4f}\n"
            f"[bold]Namespace:[/bold] {r['namespace']}\n"
            f"[bold]Title:[/bold] {r['metadata'].get('title', 'unknown')}\n"
            f"[bold]Preview:[/bold] {preview}...",
            title=f"Result #{i}",
        ))