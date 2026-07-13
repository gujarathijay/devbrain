"""
RAG Chain — connects Retriever → Prompt → LLM → Answer.

This is the core of the Q&A system. It:
  1. Takes a user question
  2. Retrieves relevant chunks from Pinecone
  3. Formats them into a prompt with citations
  4. Sends to the LLM (GPT-4o-mini)
  5. Returns the answer with sources

This module is model-agnostic — swapping GPT-4o-mini for Claude
or a local model only requires changing the LLM initialization.
"""

import openai
from rich.console import Console

from src.config import OPENAI_API_KEY
from src.retrieval.pinecone_retriever import PineconeRetriever
from src.retrieval.prompts import SYSTEM_PROMPT, format_context, build_user_prompt

console = Console()

# ── Initialize LLM client ────────────────────────────────────────
llm_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Default model — GPT-4o-mini for development (fast + cheap)
# Switch to GPT-4o for production quality
DEFAULT_MODEL = "gpt-4o-mini"


class RAGChain:
    """
    End-to-end RAG chain: question → retrieval → LLM → cited answer.

    Usage:
        chain = RAGChain()
        result = chain.ask("How do I add CORS to FastAPI?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        top_k: int = 5,
        use_mmr: bool = True,
        temperature: float = 0.1,
        use_hybrid: bool = False,
    ):
        """
        Initialize the RAG chain.

        Parameters:
        - model: which OpenAI model to use for generation
        - top_k: how many chunks to retrieve
        - use_mmr: whether to use MMR for diverse retrieval
        - temperature: LLM randomness (0.0 = deterministic, 1.0 = creative)
          We use 0.1 for RAG because we want factual, consistent answers.
          Higher temperature = more creative but more hallucination risk.
        - use_hybrid: if True, use HybridRetriever (dense + sparse + rerank)
          instead of basic PineconeRetriever. Requires Cohere API key.
        """
        self.model = model
        self.temperature = temperature

        if use_hybrid:
            from src.retrieval.hybrid_retriever import HybridRetriever
            self.retriever = HybridRetriever(top_k=top_k, use_reranker=True)
        else:
            self.retriever = PineconeRetriever(top_k=top_k, use_mmr=use_mmr)

    def ask(self, question: str) -> dict:
        """
        Ask a question and get a cited answer.

        The full pipeline:

        1. RETRIEVE: Embed the question → search Pinecone → get top_k chunks
           "How do I add CORS?" → [cors chunk, middleware chunk, ...]

        2. FORMAT: Build the prompt with context + question
           "Based on these 5 documents: [...] Answer: How do I add CORS?"

        3. GENERATE: Send prompt to LLM → get answer
           LLM reads context → writes answer with [Source: cors] citations

        4. PACKAGE: Return answer + sources + metadata for the UI

        Returns:
          {
            "answer": "To add CORS in FastAPI, use CORSMiddleware... [Source: cors]",
            "sources": [
              {"title": "cors", "url": "https://...", "score": 0.87, "namespace": "docs"},
              ...
            ],
            "question": "How do I add CORS?",
            "model": "gpt-4o-mini",
            "num_chunks_retrieved": 5,
          }
        """
        # ── Step 1: Retrieve ──────────────────────────────────────
        retrieved = self.retriever.retrieve(question)

        if not retrieved:
            return {
                "answer": "I couldn't find any relevant information in the docs or issues.",
                "sources": [],
                "question": question,
                "model": self.model,
                "num_chunks_retrieved": 0,
            }

        # ── Step 2: Format context ────────────────────────────────
        context = format_context(retrieved)
        user_prompt = build_user_prompt(context, question)

        # ── Step 3: Generate answer ───────────────────────────────
        response = llm_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content

        # ── Step 4: Package result ────────────────────────────────
        sources = []
        seen_titles = set()
        for chunk in retrieved:
            title = chunk["metadata"].get("title", "unknown")
            # Deduplicate sources (multiple chunks from same doc)
            if title not in seen_titles:
                seen_titles.add(title)
                sources.append({
                    "title": title,
                    "url": chunk["metadata"].get("url", ""),
                    "score": round(chunk["score"], 4),
                    "namespace": chunk.get("namespace", ""),
                    "source_type": chunk["metadata"].get("source", ""),
                })

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "model": self.model,
            "num_chunks_retrieved": len(retrieved),
        }


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.markdown import Markdown
    from rich.panel import Panel

    console.print("\n[bold]🧠 Testing RAG Chain[/bold]\n")

    chain = RAGChain()
    question = "How do I add CORS middleware to my FastAPI application?"

    console.print(f"[bold]Question:[/bold] {question}\n")
    console.print("[dim]Retrieving and generating...[/dim]\n")

    result = chain.ask(question)

    # Display answer
    console.print(Panel(
        Markdown(result["answer"]),
        title="DevBrain Answer",
        border_style="green",
    ))

    # Display sources
    console.print("\n[bold]📚 Sources:[/bold]")
    for s in result["sources"]:
        console.print(
            f"  • [{s['source_type']}] {s['title']} "
            f"(score: {s['score']}) — {s['url']}"
        )

    console.print(f"\n[dim]Model: {result['model']} | "
                  f"Chunks retrieved: {result['num_chunks_retrieved']}[/dim]")