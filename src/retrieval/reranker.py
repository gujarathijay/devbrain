"""
Cohere Reranker — cross-encoder reranking of candidate chunks.

Dense (Pinecone) and sparse (BM25) retrieval both score a query
against a document independently, then compare scores. A cross-encoder
reads the (query, document) pair together, so it can judge relevance
much more precisely — at the cost of being too slow to run over an
entire corpus. That's why it only reranks the small candidate set RRF
already narrowed down, rather than the whole index.
"""

from rich.console import Console

from src.config import COHERE_API_KEY

console = Console()


class CohereReranker:
    """
    Reranks candidate chunks using Cohere's rerank API.

    Usage:
        reranker = CohereReranker()
        reranked = reranker.rerank(query, candidates, top_k=5)
    """

    MODEL = "rerank-english-v3.0"

    def __init__(self):
        if COHERE_API_KEY:
            import cohere
            self.client = cohere.ClientV2(api_key=COHERE_API_KEY)
        else:
            console.print(
                "[yellow]⚠️  COHERE_API_KEY not set — reranking disabled[/yellow]"
            )
            self.client = None

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank candidate results by true relevance to the query.

        Parameters:
        - query: the user's question
        - results: candidate chunks (same shape as PineconeRetriever/BM25Retriever output)
        - top_k: how many results to return, ranked best-first

        Returns the reordered subset of `results`, each with a
        "rerank_score" field added (Cohere's relevance score, 0-1).
        """
        if not self.client or not results:
            return results[:top_k]

        documents = [r["text"] for r in results]

        response = self.client.rerank(
            model=self.MODEL,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        reranked = []
        for item in response.results:
            result = results[item.index].copy()
            result["rerank_score"] = item.relevance_score
            reranked.append(result)

        return reranked
