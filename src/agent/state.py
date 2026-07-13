"""
LangGraph State Definition.

The state is a shared data object that every node in the graph can
read from and write to. It flows through the entire pipeline:

  classify → transform → retrieve → grade → generate
     ↓           ↓           ↓         ↓         ↓
  writes      writes      writes    writes    writes
  query_type  queries     chunks    grade     answer

Think of it as a form being passed between departments:
  - Department 1 fills in "query type"
  - Department 2 fills in "transformed queries"
  - Department 3 fills in "retrieved documents"
  - etc.

Each node returns a dict of UPDATES (not the full state).
LangGraph merges the updates into the state automatically.

TypedDict vs dataclass:
  LangGraph uses TypedDict because it integrates with its
  internal state management (checkpointing, serialization).
  Each field has a type annotation for validation.
"""

from typing import TypedDict


class AgentState(TypedDict):
    """
    State that flows through the LangGraph agent.

    Every field starts as its default (empty string, empty list, 0)
    and gets populated as the graph executes.
    """

    # ── Input ─────────────────────────────────────────────────────
    question: str               # The original user question

    # ── Classification ────────────────────────────────────────────
    # What type of question is this?
    # Values: "how-to", "error", "comparison", "conceptual", "general"
    query_type: str

    # ── Query Transformation ──────────────────────────────────────
    # The transformed version(s) of the question.
    # For simple queries: 1 transformed query
    # For comparisons: 2+ decomposed sub-queries
    transformed_queries: list[str]

    # ── Retrieval ─────────────────────────────────────────────────
    # Which namespaces to search (determined by query type)
    target_namespaces: list[str]

    # All chunks retrieved across all queries and namespaces
    retrieved_chunks: list[dict]

    # ── Grading ───────────────────────────────────────────────────
    # Did retrieval find relevant documents?
    # Values: "good", "poor"
    retrieval_grade: str

    # How many times we've retried retrieval
    retry_count: int

    # Which namespaces we've already searched (to avoid repeating)
    searched_namespaces: list[str]

    # ── Generation ────────────────────────────────────────────────
    # The final answer
    answer: str

    # Source citations
    sources: list[dict]

    # ── Tracing ───────────────────────────────────────────────────
    # Log of decisions for debugging ("Classified as: how-to → Searching docs")
    trace: list[str]