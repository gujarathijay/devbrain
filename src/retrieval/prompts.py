"""
RAG Prompt Templates.

These prompts are the most important piece of the RAG pipeline.
A bad prompt = hallucinated, uncited, unreliable answers.
A good prompt = grounded, cited, helpful answers.

Prompt design principles used here:
1. ROLE: Tell the LLM who it is ("You are a FastAPI expert")
2. CONSTRAINT: Restrict it to provided context ("ONLY use the provided docs")
3. FORMAT: Tell it how to structure the answer ("Cite with [Source: ...]")
4. FALLBACK: Tell it what to do when it doesn't know ("Say 'I don't have enough info'")
5. STYLE: Tell it how to communicate ("Include code examples when relevant")

Why separate prompts into their own file?
- Easy to iterate and A/B test different prompts
- Keeps chain logic clean (rag_chain.py stays focused on orchestration)
- In Phase 5 (evaluation), we'll test different prompts against each other
"""

# ── System Prompt ─────────────────────────────────────────────────
# This is sent as the "system" message — it sets the LLM's behavior
# for the entire conversation. The LLM reads this before seeing
# the user's question or the retrieved context.

SYSTEM_PROMPT = """You are DevBrain, an expert AI assistant for FastAPI developers.

Your knowledge comes EXCLUSIVELY from the provided context documents — official
FastAPI documentation and GitHub issues. You must follow these rules strictly:

1. **Only use the provided context**: Do not use your general knowledge about FastAPI.
   If the context documents don't contain the answer, say so clearly.

2. **Cite every claim**: After each piece of information, add a citation in the format
   [Source: <title>]. Use the document title from the metadata.
   Example: "FastAPI uses Pydantic for validation [Source: tutorial/body]."

3. **Admit uncertainty**: If the context is insufficient, respond with:
   "I don't have enough information in the available docs to fully answer this.
   Here's what I found: ..." and share whatever partial information is relevant.

4. **Code examples**: When answering how-to questions, include code examples
   from the context. Format them in markdown code blocks with python syntax.

5. **Issue references**: When citing GitHub issues, include the issue number:
   [Source: Issue #1234]. This helps the developer find the original discussion.

6. **Be concise**: Answer the question directly. Don't repeat the question back
   or add unnecessary preamble."""


# ── Context Formatting ────────────────────────────────────────────
# This template formats the retrieved chunks into a structured block
# that the LLM can easily parse. Each chunk is wrapped with its
# metadata so the LLM knows WHERE each piece of information came from.

CONTEXT_TEMPLATE = """--- Document {index} ---
Source: {source}
Title: {title}
URL: {url}

{content}
"""


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context block.

    Input: list of retrieved chunk dicts from Pinecone
      [
        {"text": "## CORS...", "metadata": {"source": "docs", "title": "cors", ...}},
        ...
      ]

    Output: a single string with all chunks formatted:
      --- Document 1 ---
      Source: docs
      Title: cors
      URL: https://github.com/...

      ## CORS (Cross-Origin Resource Sharing)
      You can configure CORS in your FastAPI application...

      --- Document 2 ---
      ...

    Why this format?
    The LLM needs to know which text came from which source.
    By wrapping each chunk with clear headers ("--- Document 1 ---")
    and metadata (Source, Title, URL), the LLM can:
    - Attribute facts to specific sources
    - Generate proper citations
    - Distinguish between docs and issues
    """
    formatted_chunks = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        metadata = chunk.get("metadata", {})
        formatted = CONTEXT_TEMPLATE.format(
            index=i,
            source=metadata.get("source", "unknown"),
            title=metadata.get("title", "unknown"),
            url=metadata.get("url", ""),
            content=chunk.get("text", chunk.get("content", "")),
        )
        formatted_chunks.append(formatted)

    return "\n".join(formatted_chunks)


# ── User Prompt ───────────────────────────────────────────────────
# This template combines the formatted context with the user's question.
# It's sent as the "user" message, after the system prompt.

USER_PROMPT_TEMPLATE = """Context documents:

{context}

---

Question: {question}

Please answer based on the context documents above, with citations."""


def build_user_prompt(context: str, question: str) -> str:
    """Build the user message with context and question."""
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)