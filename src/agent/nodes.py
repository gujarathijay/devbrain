"""
Agent Node Functions — each function is one step in the LangGraph pipeline.

Every node:
  1. Receives the current state (AgentState)
  2. Does ONE thing (classify, retrieve, grade, generate, etc.)
  3. Returns a dict of state UPDATES (not the full state)

LangGraph automatically merges the returned updates into the state
before passing it to the next node.

Naming convention:
  node_classify_query  → sets query_type
  node_transform_query → sets transformed_queries
  node_retrieve        → sets retrieved_chunks
  node_grade_retrieval → sets retrieval_grade
  node_generate        → sets answer + sources
"""

import openai
from rich.console import Console

from src.config import OPENAI_API_KEY, NAMESPACE_DOCS, NAMESPACE_ISSUES
from src.agent.state import AgentState
from src.agent.query_transform import generate_hyde, decompose_query, step_back_query
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.prompts import SYSTEM_PROMPT, format_context, build_user_prompt

console = Console()

llm_client = openai.OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"

# Initialize retriever once (shared across all node calls)
_retriever = None


def _get_retriever() -> HybridRetriever:
    """Lazy-initialize the retriever (heavy object, only create once)."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(top_k=5, use_reranker=True)
    return _retriever


# ── Node 1: Classify ──────────────────────────────────────────────

def node_classify_query(state: AgentState) -> dict:
    """
    Classify the question type to determine routing strategy.

    Categories:
    - "how-to":      How do I do X? → Search docs first
    - "error":       I'm getting error X → Search issues first
    - "comparison":  Compare X vs Y → Decompose into sub-queries
    - "conceptual":  What is X? How does X work? → Search docs first
    - "general":     Anything else → Search both

    The classification determines:
    1. Which namespace to search first
    2. Which query transformation to apply
    3. How to handle the retrieved results
    """
    question = state["question"]

    response = llm_client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the following developer question into exactly one category.\n"
                    "Respond with ONLY the category name, nothing else.\n\n"
                    "Categories:\n"
                    "- how-to: questions about how to do something\n"
                    "- error: questions about errors, bugs, or things not working\n"
                    "- comparison: questions comparing two or more things\n"
                    "- conceptual: questions about what something is or how it works\n"
                    "- general: anything that doesn't fit the above"
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    query_type = response.choices[0].message.content.strip().lower()

    # Validate — fall back to "general" if LLM returns unexpected value
    valid_types = {"how-to", "error", "comparison", "conceptual", "general"}
    if query_type not in valid_types:
        query_type = "general"

    # Determine target namespaces based on query type
    namespace_routing = {
        "how-to": [NAMESPACE_DOCS],
        "error": [NAMESPACE_ISSUES, NAMESPACE_DOCS],
        "comparison": [NAMESPACE_DOCS],
        "conceptual": [NAMESPACE_DOCS],
        "general": [NAMESPACE_DOCS, NAMESPACE_ISSUES],
    }

    target_ns = namespace_routing.get(query_type, [NAMESPACE_DOCS, NAMESPACE_ISSUES])

    trace_msg = f"Classified as: {query_type} → Targeting: {', '.join(target_ns)}"
    console.print(f"  [dim]🏷️  {trace_msg}[/dim]")

    return {
        "query_type": query_type,
        "target_namespaces": target_ns,
        "trace": state.get("trace", []) + [trace_msg],
    }


# ── Node 2: Transform ────────────────────────────────────────────

def node_transform_query(state: AgentState) -> dict:
    """
    Transform the query based on its type for better retrieval.

    Routing logic:
    - comparison → decompose into sub-queries
    - error → use original + step-back (broader context helps)
    - how-to → use HyDE (hypothetical doc matches real docs)
    - conceptual → use HyDE
    - general → use original query as-is
    """
    question = state["question"]
    query_type = state["query_type"]

    if query_type == "comparison":
        # Decompose: "Compare X vs Y" → ["What is X?", "What is Y?"]
        queries = decompose_query(question)
        transform_name = "decomposition"

    elif query_type == "error":
        # For errors: search with original (for exact error match) +
        # step-back (for broader context about the system)
        stepped = step_back_query(question)
        queries = [question, stepped]
        transform_name = "original + step-back"

    elif query_type in ("how-to", "conceptual"):
        # HyDE: generate a hypothetical doc excerpt, search with that
        hyde = generate_hyde(question)
        # Include original too — HyDE is additive, not a replacement
        queries = [question, hyde]
        transform_name = "original + HyDE"

    else:
        # General: just use the original
        queries = [question]
        transform_name = "passthrough"

    trace_msg = f"Transform: {transform_name} → {len(queries)} queries"
    console.print(f"  [dim]🔄 {trace_msg}[/dim]")

    return {
        "transformed_queries": queries,
        "trace": state.get("trace", []) + [trace_msg],
    }


# ── Node 3: Retrieve ─────────────────────────────────────────────

def node_retrieve(state: AgentState) -> dict:
    """
    Retrieve relevant chunks using the transformed queries.

    For each transformed query, search the target namespace(s).
    Merge all results and deduplicate.

    If this is a RETRY (retry_count > 0), we search namespaces
    we haven't tried yet — that's the corrective part of Corrective RAG.
    """
    queries = state["transformed_queries"]
    searched = state.get("searched_namespaces", [])
    retry_count = state.get("retry_count", 0)
    retriever = _get_retriever()

    # On retry, expand to all namespaces we haven't searched yet
    if retry_count > 0:
        all_namespaces = [NAMESPACE_DOCS, NAMESPACE_ISSUES]
        target_ns = [ns for ns in all_namespaces if ns not in searched]
        if not target_ns:
            # We've searched everywhere — use all namespaces as last resort
            target_ns = all_namespaces
        trace_msg = f"RETRY #{retry_count}: Expanding search to: {', '.join(target_ns)}"
    else:
        target_ns = state.get("target_namespaces", [NAMESPACE_DOCS])
        trace_msg = f"Searching: {', '.join(target_ns)} with {len(queries)} queries"

    console.print(f"  [dim]🔍 {trace_msg}[/dim]")

    # Retrieve for each query
    all_chunks = []
    seen_texts = set()

    for query in queries:
        results = retriever.retrieve(query)

        for chunk in results:
            # Deduplicate by first 200 chars of text
            text_key = chunk["text"][:200]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_chunks.append(chunk)

    # Combine with any existing chunks from previous attempts
    existing = state.get("retrieved_chunks", [])
    for chunk in existing:
        text_key = chunk["text"][:200]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            all_chunks.append(chunk)

    # Sort by best available score
    def _best_score(chunk):
        return chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))

    all_chunks.sort(key=_best_score, reverse=True)

    # Keep top 10 to avoid context window overflow
    all_chunks = all_chunks[:10]

    console.print(f"  [dim]📄 Retrieved {len(all_chunks)} unique chunks[/dim]")

    return {
        "retrieved_chunks": all_chunks,
        "searched_namespaces": searched + target_ns,
        "trace": state.get("trace", []) + [trace_msg],
    }


# ── Node 4: Grade Retrieval ──────────────────────────────────────

def node_grade_retrieval(state: AgentState) -> dict:
    """
    Grade whether the retrieved documents can answer the question.

    This is the CORRECTIVE part of Corrective RAG. We ask the LLM:
    "Given these documents, can you answer this question?"

    If YES → proceed to generate answer
    If NO  → retry with different namespaces/strategy

    Why use the LLM for grading instead of a simple score threshold?
    Because cosine similarity only measures topic similarity, not
    whether the document actually ANSWERS the question. A doc about
    CORS (topic match) might not answer "How to configure CORS with
    specific headers" (too general).
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]
    retry_count = state.get("retry_count", 0)

    # If no chunks at all, it's definitely poor
    if not chunks:
        console.print("  [dim]📊 Grade: POOR (no chunks retrieved)[/dim]")
        return {
            "retrieval_grade": "poor",
            "trace": state.get("trace", []) + ["Grade: POOR — no chunks"],
        }

    # If we've already retried twice, accept what we have
    if retry_count >= 2:
        console.print("  [dim]📊 Grade: ACCEPTED (max retries reached)[/dim]")
        return {
            "retrieval_grade": "good",
            "trace": state.get("trace", []) + ["Grade: ACCEPTED — max retries"],
        }

    # Build a summary of retrieved content for the grader
    chunk_summaries = []
    for i, chunk in enumerate(chunks[:5], 1):
        preview = chunk["text"][:300]
        chunk_summaries.append(f"Document {i}: {preview}")

    docs_text = "\n\n".join(chunk_summaries)

    response = llm_client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a retrieval quality grader. Given a question and "
                    "retrieved documents, determine if the documents contain "
                    "enough information to answer the question.\n\n"
                    "Respond with ONLY 'yes' or 'no'.\n"
                    "- 'yes': documents contain relevant information to answer\n"
                    "- 'no': documents are off-topic or too generic to answer"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nRetrieved documents:\n{docs_text}",
            },
        ],
    )

    grade_response = response.choices[0].message.content.strip().lower()
    grade = "good" if "yes" in grade_response else "poor"

    trace_msg = f"Grade: {grade.upper()}"
    console.print(f"  [dim]📊 {trace_msg}[/dim]")

    return {
        "retrieval_grade": grade,
        "trace": state.get("trace", []) + [trace_msg],
    }


# ── Node 5: Retry (Corrective RAG) ───────────────────────────────

def node_retry(state: AgentState) -> dict:
    """
    Prepare for a retry with different search strategy.

    This simply increments the retry counter.
    The retrieve node checks retry_count and adjusts its behavior:
    - retry_count=0: search target namespaces
    - retry_count=1: expand to all unsearched namespaces
    - retry_count=2: last attempt with everything
    """
    retry_count = state.get("retry_count", 0) + 1
    trace_msg = f"Retrying retrieval (attempt {retry_count})"
    console.print(f"  [dim]🔁 {trace_msg}[/dim]")

    return {
        "retry_count": retry_count,
        "trace": state.get("trace", []) + [trace_msg],
    }


# ── Node 6: Generate ─────────────────────────────────────────────

def node_generate(state: AgentState) -> dict:
    """
    Generate the final answer using retrieved context.

    This is the same generation logic from Phase 2's RAG chain,
    but now with better-quality retrieved chunks (thanks to
    classification, transformation, and grading).
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    # Format context and build prompt
    context = format_context(chunks)
    user_prompt = build_user_prompt(context, question)

    response = llm_client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content

    # Build source list (deduplicated)
    sources = []
    seen_titles = set()
    for chunk in chunks:
        title = chunk["metadata"].get("title", "unknown")
        if title not in seen_titles:
            seen_titles.add(title)
            sources.append({
                "title": title,
                "url": chunk["metadata"].get("url", ""),
                "source_type": chunk["metadata"].get("source", ""),
            })

    trace_msg = f"Generated answer ({len(answer)} chars, {len(sources)} sources)"
    console.print(f"  [dim]✍️  {trace_msg}[/dim]")

    return {
        "answer": answer,
        "sources": sources,
        "trace": state.get("trace", []) + [trace_msg],
    }