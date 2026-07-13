"""
LangGraph Agent — builds and compiles the graph.

This module defines the graph structure — which nodes exist,
how they're connected, and what conditions control the flow.

The graph looks like:

    START
      │
      ▼
  classify_query
      │
      ▼
  transform_query
      │
      ▼
  retrieve ◄────────┐
      │              │
      ▼              │
  grade_retrieval    │
      │              │
      ├─── GOOD ───▶ generate ──▶ END
      │              │
      └─── POOR ───▶ retry ──────┘
                   (max 2 retries)

LangGraph compiles this into an executable function that takes
an initial state and returns the final state after all nodes run.
"""

from langgraph.graph import END, StateGraph

from src.agent.state import AgentState
from src.agent.nodes import (
    node_classify_query,
    node_transform_query,
    node_retrieve,
    node_grade_retrieval,
    node_retry,
    node_generate,
)


def _should_retry_or_generate(state: AgentState) -> str:
    """
    Conditional edge: after grading, decide whether to generate or retry.

    This is the DECISION POINT of Corrective RAG:
    - If retrieval quality is "good" → proceed to generate the answer
    - If retrieval quality is "poor" → retry with different strategy

    Returns the name of the next node to execute.
    """
    if state["retrieval_grade"] == "good":
        return "generate"
    else:
        return "retry"


def build_graph() -> StateGraph:
    """
    Build the LangGraph agent graph.

    Each add_node(name, function) registers a node.
    Each add_edge(from, to) connects nodes unconditionally.
    add_conditional_edges(from, condition_fn, mapping) adds a branch.

    The mapping dict maps the condition function's return values
    to node names: {"generate": "generate", "retry": "retry"}
    """
    # Create the graph with our state type
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("classify", node_classify_query)
    graph.add_node("transform", node_transform_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade", node_grade_retrieval)
    graph.add_node("retry", node_retry)
    graph.add_node("generate", node_generate)

    # ── Define edges ──────────────────────────────────────────────

    # Start → classify (always)
    graph.set_entry_point("classify")

    # classify → transform (always)
    graph.add_edge("classify", "transform")

    # transform → retrieve (always)
    graph.add_edge("transform", "retrieve")

    # retrieve → grade (always)
    graph.add_edge("retrieve", "grade")

    # grade → generate OR retry (conditional)
    graph.add_conditional_edges(
        "grade",
        _should_retry_or_generate,
        {
            "generate": "generate",
            "retry": "retry",
        },
    )

    # retry → retrieve (loop back)
    graph.add_edge("retry", "retrieve")

    # generate → END (always)
    graph.add_edge("generate", END)

    return graph


def create_agent():
    """
    Build and compile the graph into an executable agent.

    compile() turns the graph definition into a runnable function.
    After compilation, you can call:

        result = agent.invoke({"question": "How do I add CORS?"})
        print(result["answer"])

    The agent executes all nodes in order, following edges and
    conditions, and returns the final state.
    """
    graph = build_graph()
    agent = graph.compile()
    return agent


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()

    console.print("\n[bold]🤖 Testing LangGraph Agent[/bold]\n")

    agent = create_agent()

    question = "I'm getting a 422 error when sending a POST request with nested Pydantic models"

    console.print(f"[bold]Question:[/bold] {question}\n")
    console.print("[bold]Agent trace:[/bold]")

    # Run the agent
    initial_state = {
        "question": question,
        "query_type": "",
        "transformed_queries": [],
        "target_namespaces": [],
        "retrieved_chunks": [],
        "retrieval_grade": "",
        "retry_count": 0,
        "searched_namespaces": [],
        "answer": "",
        "sources": [],
        "trace": [],
    }

    result = agent.invoke(initial_state)

    # Display trace
    console.print("\n[bold]Full trace:[/bold]")
    for step in result["trace"]:
        console.print(f"  → {step}")

    # Display answer
    console.print()
    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]DevBrain Agent Answer[/bold green]",
        border_style="green",
    ))

    # Display sources
    console.print("\n[bold]📚 Sources:[/bold]")
    for s in result["sources"]:
        icon = "📄" if s["source_type"] == "docs" else "🐛"
        console.print(f"  {icon} {s['title']} — {s['url']}")